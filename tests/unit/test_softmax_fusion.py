"""Regression tests for the softmax_fusion branch: the attention softmax is the fused
ops.softmax for every dtype. The op-by-op softmax made the negated, broadcast max a
full (batch, heads, T, T) constant -- a leaf, which gradient checkpointing cannot
release -- and kept logit - max, the exp and the quotient for backward."""
import gc
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import ops
from needle.autograd import Tensor, find_topo_sort
from needle.backend_ndarray import ndarray as nd
from needle.nn import nn_transformer
from needle.nn.nn_transformer import MultiHeadAttention

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f32 = np.float32
B, H, T, E = 4, 8, 64, 64


def op_by_op_softmax(self, logit):
    """The attention softmax before this branch (for comparison)."""
    max_val = Tensor(logit.realize_cached_data().max(axis=3), device=logit.device, dtype=logit.dtype,
                     requires_grad=False)
    max_val = max_val.reshape((*logit.shape[:-1], 1)).broadcast_to(logit.shape)
    probs = ops.exp(logit - max_val)
    denom = probs.sum(axes=3).reshape((*logit.shape[:-1], 1)).broadcast_to(logit.shape)
    return probs / denom


def np_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def make_lm(device, policies, layers=2, half_softmax=False):
    from models import LanguageModel

    np.random.seed(0)
    return LanguageModel(E, 100, 256, num_layers=layers, seq_model="transformer", seq_len=T, device=device,
                         policies=policies, half_softmax=half_softmax)


def batch(device, seed=1):
    rng = np.random.RandomState(seed)
    return (ndl.Tensor(rng.randint(0, 100, (T, B)), device=device, requires_grad=False),
            ndl.Tensor(rng.randint(0, 100, T * B), device=device, requires_grad=False))


# --------------------------------------------------------------------------- values


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_attention_softmax_values_and_gradient(device):
    """With a causal mask (large negative logits), the fused softmax gives the
    reference values and gradient, and agrees with the op-by-op version."""
    rng = np.random.RandomState(0)
    x = (2 * rng.randn(2, 3, 16, 16)).astype(f32)
    x += np.triu(np.ones((16, 16), f32), 1) * MultiHeadAttention.causal_mask_value()
    g = rng.randn(2, 3, 16, 16).astype(f32)
    attn = MultiHeadAttention(causal=True, device=device)
    res = {}
    for name, fn in (("fused", MultiHeadAttention.softmax), ("op-by-op", op_by_op_softmax)):
        t = ndl.Tensor(x, device=device, requires_grad=True)
        p = fn(attn, t)
        (p * ndl.Tensor(g, device=device)).sum().backward()
        res[name] = (p.numpy(), t.grad.numpy())
    ref = np_softmax(x.astype(np.float64))
    ref_grad = ref * (g - (g * ref).sum(-1, keepdims=True))
    np.testing.assert_allclose(res["fused"][0], ref, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(res["fused"][1], ref_grad, rtol=1e-4, atol=1e-6)
    for a, b in zip(res["fused"], res["op-by-op"]):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_model_results_close_to_the_op_by_op_softmax(device, monkeypatch):
    x, y = batch(device)
    res = {}
    for name in ("fused", "op-by-op"):
        if name == "op-by-op":
            monkeypatch.setattr(MultiHeadAttention, "softmax", op_by_op_softmax)
        m = make_lm(device, ["fp32", "none"])
        loss = nn.SoftmaxLoss()(m(x)[0], y)
        loss.backward()
        res[name] = [loss.numpy()] + [p.grad.numpy() for p in m.parameters()]
    for a, b in zip(res["fused"], res["op-by-op"]):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_half_softmax_changes_only_the_backward():
    """Without half_softmax the float32 softmax is rounded to float16 by the next
    matmul's autocast cast -- the rounding of the fused float16 softmax: the loss is
    the same, the gradients differ."""
    dev = ndl.cpu()
    x, y = batch(dev)
    res = {}
    for flag in (False, True):
        m = make_lm(dev, "fp16", half_softmax=flag)
        loss = nn.SoftmaxLoss()(m(x)[0], y)
        loss.backward()
        res[flag] = (loss.numpy(), [p.grad.numpy() for p in m.parameters()])
    np.testing.assert_array_equal(res[False][0], res[True][0])
    assert any(not np.array_equal(a, b) for a, b in zip(res[False][1], res[True][1]))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_model_softmax_gradient_is_safe(device):
    """A float16 model without autocast now uses the fused softmax too: the gradient
    of long, nearly flat rows (row sums near 512) is accurate. (The op-by-op softmax
    squared the row sum in its division's gradient: inf in float16.)"""
    rng = np.random.RandomState(2)
    x = (0.01 * rng.randn(1, 1, 4, 512)).astype(f32)
    g = rng.randn(1, 1, 4, 512).astype(f32)
    t = ndl.Tensor(x, device=device, dtype="float16", requires_grad=True)
    p = MultiHeadAttention(device=device, dtype="float16").softmax(t)
    assert p.dtype == "float16"
    (p * ndl.Tensor(g, device=device, dtype="float16")).sum().backward()
    ref = np_softmax(x.astype(np.float64))
    ref_grad = ref * (g - (g * ref).sum(-1, keepdims=True))
    grad = t.grad.numpy().astype(f32)
    assert np.all(np.isfinite(grad)) and np.abs(grad - ref_grad).max() <= 2e-2 * np.abs(ref_grad).max()


# --------------------------------------------------------------------------- memory


def _segment_nodes(m, x):
    seen = []
    m.sequence_model.segment_hook = lambda ev, i, p, t: seen.append(t)
    loss = nn.SoftmaxLoss()(m(x)[0], batch(x.device)[1])
    m.sequence_model.segment_hook = None
    nodes = find_topo_sort([loss])
    return loss, nodes[nodes.index(seen[2]) + 1:nodes.index(seen[3]) + 1]  # segment 1


def test_checkpointed_layer_keeps_no_attention_sized_constant():
    dev = ndl.cpu()
    x, _ = batch(dev)
    loss, inside = _segment_nodes(make_lm(dev, "none"), x)
    big = B * H * T * T
    for v in inside:
        d = v.cached_data
        if d is not None and not v.requires_grad:  # constants made inside the layer
            assert d._handle.size < big, (tuple(d.shape), d.strides)  # views of small buffers only
    kept = [v for v in inside if v.cached_data is not None and v.op is not None]
    assert len(kept) == 1 and tuple(kept[0].shape) == (B, T, E)  # the layer's output, nothing else


@pytest.mark.parametrize("policy,saved", [("none", 1), ("fp32", 2)])
def test_layer_memory_drops(policy, saved, monkeypatch):
    """After its forward a checkpointed layer holds one attention-sized tensor less
    (the constant), a float32 layer at least two less (the constant and the op-by-op
    intermediates)."""
    dev = ndl.cpu()
    x, y = batch(dev)

    def held():
        m = make_lm(dev, policy, layers=1)
        marks = {}
        m.sequence_model.segment_hook = lambda ev, i, p, t: marks.__setitem__(ev, nd.memory_allocated(dev))
        loss = nn.SoftmaxLoss()(m(x)[0], y)
        out = marks["end"] - marks["start"]
        del loss, m
        gc.collect()
        return out

    fused = held()
    monkeypatch.setattr(MultiHeadAttention, "softmax", op_by_op_softmax)
    old = held()
    attention_tensor = B * H * T * T * 4
    assert old - fused >= saved * attention_tensor, (old, fused)
    if policy == "none":
        assert fused < attention_tensor  # only the output (and small constants) is left
