"""Switchable per-layer segment policies ("fp32" / "fp16" / "none") in nn.Transformer."""
import gc
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import autograd, ops
from needle.backend_ndarray import ndarray as nd
from needle.nn.nn_transformer import SEGMENT_POLICIES, run_segment

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f32 = np.float32
LAYERS = 3


def make_lm(device, policies=None, checkpoint=False, dropout=0.0, layers=LAYERS, size=(16, 30, 32), **kw):
    from models import LanguageModel

    emb, vocab, hidden = size
    np.random.seed(0)
    m = LanguageModel(emb, vocab, hidden, num_layers=layers, seq_model="transformer", device=device,
                      checkpoint=checkpoint, policies=policies)
    for layer in m.sequence_model.layers.modules:
        layer.dropout.p = dropout
        layer.attn_layer.attn.dropout.p = dropout
    m.sequence_model.fp16_release_inputs = kw.get("fp16_release_inputs", True)
    return m


def batch(device, seed=1, vocab=30, shape=(8, 3)):
    rng = np.random.RandomState(seed)
    x = ndl.Tensor(rng.randint(0, vocab, shape), device=device, requires_grad=False)
    y = ndl.Tensor(rng.randint(0, vocab, shape[0] * shape[1]), device=device, requires_grad=False)
    return x, y


def manual_forward(m, x, policies, release_inputs=True):
    """Independent reference for LanguageModel.forward with segment policies."""
    tr = m.sequence_model
    seq_len, bs = x.shape
    h = m.embedding_layer(x)
    h = ops.transpose(h, axes=(0, 1))
    pos_ids = ndl.Tensor(np.arange(seq_len, dtype=f32), device=x.device, requires_grad=False)
    pos_ids = ops.broadcast_to(ops.reshape(pos_ids, (1, seq_len)), (bs, seq_len))
    h = h + tr.pos_embed(pos_ids)
    for layer, p in zip(tr.layers.modules, policies):
        h = h.astype("float32", release_input=False)
        if p == "fp16":
            with ndl.autocast(dtype="float16", release_inputs=release_inputs):
                h = layer(h)
        elif p == "none":
            with ndl.autocast(enabled=False):
                h = ndl.checkpoint(layer, h)
        else:
            with ndl.autocast(enabled=False):
                h = layer(h)
        h = h.astype("float32", release_input=False)
    h = ops.transpose(h, axes=(0, 1))
    out = ops.reshape(h, (seq_len * bs, m.embedding_size))
    return m.final_proj(out)


def loss_and_grads(m, x, y, forward=None, amp=None):
    np.random.seed(7)  # dropout masks
    with ndl.autocast(enabled=amp is not None, dtype=amp or "float16"):
        out = forward(m, x) if forward else m(x)[0]
        loss = nn.SoftmaxLoss()(out, y)
    loss.backward()
    return loss.numpy(), [p.grad.numpy() for p in m.parameters()]


def assert_same(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    for g, h in zip(a[1], b[1]):
        np.testing.assert_array_equal(g, h)


def record_modes(m):
    """Wrap every layer to record (autocast enabled, autocast dtype) per call."""
    seen = []
    for i, layer in enumerate(m.sequence_model.layers.modules):
        fwd = layer.forward

        def spy(x, _fwd=fwd, _i=i):
            seen.append((_i, ndl.is_autocast_enabled(), ndl.get_autocast_dtype(), x.dtype))
            return _fwd(x)

        layer.forward = spy
    return seen


# --------------------------------------------------------------------------- API


def test_policy_validation():
    m = make_lm(ndl.cpu())
    tr = m.sequence_model
    assert tr.policies is None and SEGMENT_POLICIES == ("fp32", "fp16", "none")
    tr.set_policies("fp16")
    assert tr.policies == ["fp16"] * LAYERS
    tr.set_policies(("none", "fp32", "fp16"))
    assert tr.policies == ["none", "fp32", "fp16"]
    for bad in (["fp32"] * (LAYERS - 1), ["fp32", "fp32", "bf16"], "int8"):
        with pytest.raises(ValueError):
            tr.set_policies(bad)
    assert tr.policies == ["none", "fp32", "fp16"]  # unchanged after a rejected call
    tr.set_policies(None)
    assert tr.policies is None
    with pytest.raises(ValueError, match="unknown segment policy"):
        run_segment(lambda v: v, ndl.Tensor(np.ones(2, f32)), "fp8")


def test_float32_model_required():
    np.random.seed(0)
    tr = nn.Transformer(16, 32, 2, num_head=2, dim_head=8, device=ndl.cpu(), dtype="float16")
    with pytest.raises(ValueError, match="float32 model"):
        tr.set_policies("fp32")


def test_language_model_set_policies_and_scaling_flag():
    m = make_lm(ndl.cpu(), policies="fp32")
    assert not m.sequence_model.needs_loss_scaling
    m.set_policies(["fp32", "fp16", "none"])
    assert m.sequence_model.policies == ["fp32", "fp16", "none"]
    assert m.sequence_model.needs_loss_scaling
    from models import LanguageModel

    lstm = LanguageModel(8, 20, 16, seq_model="lstm", device=ndl.cpu())
    with pytest.raises(ValueError, match="transformer"):
        lstm.set_policies("fp32")


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_fp32_and_none_bit_identical_to_plain_model(device):
    x, y = batch(device)
    ref = loss_and_grads(make_lm(device, dropout=0.1), x, y)
    for pol in ("fp32", "none", ["none", "fp32", "none"]):
        assert_same(loss_and_grads(make_lm(device, policies=pol, dropout=0.1), x, y), ref)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_fp16_and_mixed_match_manual_reference(device):
    x, y = batch(device)
    for pol in (["fp16"] * LAYERS, ["fp32", "fp16", "none"], ["fp16", "none", "fp16"]):
        for release in (True, False):
            got = loss_and_grads(make_lm(device, policies=pol, fp16_release_inputs=release), x, y)
            ref = loss_and_grads(make_lm(device), x, y,
                                 forward=lambda m, x: manual_forward(m, x, pol, release))
            assert_same(got, ref)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_precision_per_segment(device):
    m = make_lm(device, policies=["fp32", "fp16", "none"])
    seen = record_modes(m)
    x, y = batch(device)
    loss_and_grads(m, x, y)
    # the "none" layer runs once in forward (backward replays recorded ops, not fn)
    assert [(i, on, dt if on else None, xd) for i, on, dt, xd in seen] == [
        (0, False, None, "float32"), (1, True, "float16", "float32"), (2, False, None, "float32")]


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_policy_overrides_enclosing_autocast(device):
    """Inside autocast(float16 / bfloat16) the segments still follow their policy,
    and their inputs are float32 even though the embedding ran in half precision."""
    for amp in ("float16", "bfloat16"):
        m = make_lm(device, policies=["fp32", "fp16", "none"])
        seen = record_modes(m)
        x, y = batch(device)
        _, grads = loss_and_grads(m, x, y, amp=amp)
        assert [(on, dt if on else None, xd) for _, on, dt, xd in seen] == [
            (False, None, "float32"), (True, "float16", "float32"), (False, None, "float32")]
        assert all(np.isfinite(g).all() for g in grads)
    # "fp32" and "none" segments compute identically inside the enclosing region
    x, y = batch(device)
    assert_same(loss_and_grads(make_lm(device, policies="fp32"), x, y, amp="float16"),
                loss_and_grads(make_lm(device, policies="none"), x, y, amp="float16"))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_boundaries_float32_kept_and_interior_only_for_none(device):
    x, y = batch(device)
    counts = {}
    for pol in ("fp32", "fp16", "none", ["none", "fp16", "fp32"]):
        m = make_lm(device, policies=pol)
        out = m(x)[0]
        nodes = autograd.find_topo_sort([out])
        boundaries = [n for n in nodes if n._keep_cached]
        assert len(boundaries) >= LAYERS + 1
        assert all(n.dtype == "float32" and n.cached_data is not None for n in boundaries)
        counts[str(pol)] = sum(n._checkpoint_interior for n in nodes)
    assert counts["fp32"] == 0 and counts["fp16"] == 0 and counts["none"] > 0
    assert counts[str(["none", "fp16", "fp32"])] * LAYERS == counts["none"]  # one of three layers


def test_memory_ordering():
    dev = ndl.cpu()
    x, y = batch(dev, vocab=100, shape=(32, 8))
    held = {}
    for key, pol, release in (("fp32", "fp32", True), ("fp16", "fp16", True),
                              ("fp16-keep-inputs", "fp16", False), ("none", "none", True)):
        m = make_lm(dev, policies=pol, size=(64, 100, 256), fp16_release_inputs=release)
        gc.collect()
        base = nd.memory_allocated(dev)
        out = m(x)[0]
        held[key] = nd.memory_allocated(dev) - base
        del out, m
        gc.collect()
    assert held["none"] < held["fp16"] < held["fp32"]
    assert held["fp16"] < held["fp16-keep-inputs"]


# --------------------------------------------------------------------------- switching during training


SCHEDULE = [["fp32"] * 3, ["fp16", "none", "fp32"], ["none"] * 3, ["fp16"] * 3, ["fp32", "fp32", "fp16"]]


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_switching_between_steps_matches_reference(device):
    """Policies changed every step: the same parameters as the manual reference
    that applies the same schedule, with one GradScaler kept across switches."""
    results = []
    for use_model_policies in (True, False):
        m = make_lm(device, policies=SCHEDULE[0])
        opt = ndl.optim.Adam(m.parameters(), lr=1e-2)
        sc = ndl.amp.GradScaler(init_scale=2.0 ** 8)
        scales = []
        for step, pol in enumerate(SCHEDULE):
            x, y = batch(device, seed=10 + step)
            opt.reset_grad()
            np.random.seed(step)
            if use_model_policies:
                m.set_policies(pol)
                out = m(x)[0]
            else:
                out = manual_forward(m, x, pol)
            loss = nn.SoftmaxLoss()(out, y)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
            scales.append(sc.get_scale())
        results.append(([p.numpy() for p in m.parameters()], scales))
    for a, b in zip(results[0][0], results[1][0]):
        np.testing.assert_array_equal(a, b)
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_switching_only_exact_policies_equals_fp32_training(device):
    """Steps alternating "fp32" / "none" / mixes of both train exactly like fp32."""
    params = []
    for switching in (False, True):
        m = make_lm(device, policies="fp32", dropout=0.1)
        opt = ndl.optim.SGD(m.parameters(), lr=0.1)
        for step in range(4):
            if switching:
                m.set_policies([["none"] * 3, ["fp32", "none", "fp32"], "fp32", ["none", "none", "fp32"]][step])
            x, y = batch(device, seed=20 + step)
            opt.reset_grad()
            np.random.seed(step)
            nn.SoftmaxLoss()(m(x)[0], y).backward()
            opt.step()
        params.append([p.numpy() for p in m.parameters()])
    for a, b in zip(*params):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_evaluation_under_no_grad(device):
    x, _ = batch(device)
    outs = []
    for pol in ("fp32", "none", ["none", "fp32", "fp32"]):
        m = make_lm(device, policies=pol)
        m.eval()
        with ndl.no_grad():
            out = m(x)[0]
        assert not out.requires_grad
        outs.append(out.numpy())
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


# --------------------------------------------------------------------------- boundary protection


class _Counted(autograd.TensorOp):
    """Identity that counts forward evaluations (to detect recomputation)."""

    n = 0

    def compute(self, a):
        _Counted.n += 1
        return a * 1.0

    def gradient(self, out_grad, node):
        return out_grad


def test_casts_do_not_release_segment_boundaries():
    """Residual segments v + f(cast(v)): a cast that releases its input (autocast
    release_inputs=True, or an explicit .half()) must not release the previous
    segment's output, or reading it again recomputes that whole segment in forward."""
    dev = ndl.cpu()
    np.random.seed(1)
    layers = [nn.Linear(64, 64, device=dev) for _ in range(4)]
    x = ndl.Tensor(np.random.randn(32, 64).astype(f32), device=dev)

    def block(l, explicit):
        def f(v):
            # explicit: casts written by hand (works with autocast on or off)
            h = ops.relu(v.half() @ l.weight.half()) if explicit else ops.relu(l(v))
            return v + _Counted()(h.float())
        return f

    for policy in ("fp16", "none"):
        for explicit in (False, True):
            _Counted.n = 0
            h = x
            for l in layers:
                h = run_segment(block(l, explicit), h, policy, release_inputs=True)
            loss = ops.summation(h * h)
            assert _Counted.n == len(layers)  # no segment recomputed during forward
            interior = [n for n in autograd.find_topo_sort([loss]) if n._checkpoint_interior]
            assert all(n.cached_data is None for n in interior)
            loss.backward()
            assert all(p.grad is not None for l in layers for p in l.parameters())
