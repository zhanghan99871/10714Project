"""autocast(dtype="bfloat16"), the bfloat16 causal mask and amp="bfloat16" in simple_ml."""
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
import needle as ndl
import needle.nn as nn
from needle import autograd, ops

bf16 = ml_dtypes.bfloat16
f16, f32 = np.float16, np.float32
_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
DEV = ndl.cpu()


def T(x, device, dtype=None, requires_grad=True):
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def bits(x):
    return np.ascontiguousarray(x).view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def mlp(device, seed=0):
    np.random.seed(seed)
    return nn.Sequential(nn.Linear(8, 32, device=device), nn.ReLU(), nn.Linear(32, 4, device=device))


def batch(device, seed=1):
    np.random.seed(seed)
    return (T(np.random.randn(16, 8).astype(f32), device, requires_grad=False),
            T(np.random.randint(0, 4, 16), device, requires_grad=False))


def flat_grads(model):
    return np.concatenate([p.grad.numpy().astype(np.float64).ravel() for p in model.parameters()])


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# --------------------------------------------------------------------------- autocast(dtype="bfloat16")


def test_autocast_bf16_state():
    with ndl.autocast(dtype="bfloat16"):
        assert ndl.get_autocast_dtype() == "bfloat16"
        with ndl.autocast():  # nested float16 region
            assert ndl.get_autocast_dtype() == "float16"
        assert ndl.get_autocast_dtype() == "bfloat16"
    with ndl.autocast(dtype=bf16):
        assert ndl.get_autocast_dtype() == "bfloat16"
    assert not ndl.is_autocast_enabled()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_policies(device):
    np.random.seed(0)
    a = T(np.random.randn(4, 5).astype(f32), device)
    w = T(np.random.randn(5, 3).astype(f32), device)
    x = T(np.random.randn(1, 6, 6, 2).astype(f32), device)
    k = T(np.random.randn(3, 3, 2, 4).astype(f32), device)
    hb = T(np.random.uniform(0.5, 2, (3, 4)).astype(bf16), device)
    hh = T(np.random.uniform(0.5, 2, (3, 4)).astype(f16), device)
    with ndl.autocast(dtype="bfloat16"):
        assert (a @ w).dtype == "bfloat16"
        assert ops.conv(x, k, padding=1).dtype == "bfloat16"
        assert (ops.reshape(a, (1, 4, 5)) @ w).dtype == "bfloat16"  # batched matmul too
        for y in (ops.summation(hb), ops.exp(hb), ops.log(hb), ops.logsoftmax(hb), hb ** 2):
            assert y.dtype == "float32"
        assert (hb + hb).dtype == "bfloat16" and ops.relu(hb).dtype == "bfloat16"
        assert (hb + hh).dtype == "float32"  # float16 + bfloat16 -> float32
        assert (hb * T(np.ones((3, 4), f32), device)).dtype == "float32"
        assert ops.stack([hb, hh], 0).dtype == "float32"
        assert (hh @ ops.transpose(hb)).dtype == "bfloat16"  # "lower" casts both to bf16
    np.testing.assert_array_equal(
        bits(ops.matmul(a.bfloat16(), w.bfloat16()).numpy()),
        bits(_autocast_matmul(a, w)),
    )


def _autocast_matmul(a, w):
    with ndl.autocast(dtype="bfloat16"):
        return (a @ w).numpy()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_autocast_equals_manual_casts(device):
    res = {}
    for mode in ("auto", "manual"):
        model, (X, y) = mlp(device), batch(device)
        l1, l2 = model.modules[0], model.modules[2]
        if mode == "auto":
            with ndl.autocast(dtype="bfloat16"):
                loss = nn.SoftmaxLoss()(model(X), y)
        else:
            h = (X.bfloat16() @ l1.weight.bfloat16()).float()
            h = ops.relu(h + ops.broadcast_to(ops.reshape(l1.bias, (1, 32)), (16, 32)))
            o = (h.bfloat16() @ l2.weight.bfloat16()).float()
            loss = nn.SoftmaxLoss()(o + ops.broadcast_to(ops.reshape(l2.bias, (1, 4)), (16, 4)), y)
        loss.backward()
        assert all(p.grad.dtype == "float32" for p in model.parameters())
        res[mode] = [loss.numpy()] + [p.grad.numpy() for p in model.parameters()]
    for a, b in zip(res["auto"], res["manual"]):
        np.testing.assert_array_equal(bits(a), bits(b))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_parameter_cast_cache_per_dtype(device):
    W = T(np.ones((3, 3), f32), device)
    x = T(np.ones((2, 3), f32), device, requires_grad=False)
    with ndl.autocast(dtype="bfloat16"):
        a = x @ W
        with ndl.autocast(dtype="float16"):
            b = x @ W
        c = x @ W
    assert a.dtype == c.dtype == "bfloat16" and b.dtype == "float16"
    casts = [n for n in autograd.find_topo_sort([ops.summation(a) + ops.summation(c)])
             if isinstance(n.op, ops.Cast) and n.inputs[0] is W]
    assert len(casts) == 1  # the bfloat16 cast is reused within the region


# --------------------------------------------------------------------------- no loss scaling needed


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_tiny_gradients_survive_without_scaling(device):
    """dloss/dW = 3e-8: lost in float16 without loss scaling, fine in bfloat16."""
    res = {}
    for dt in ("float16", "bfloat16"):
        W = T(np.ones((4, 2), f32), device)
        x = T(np.full((3, 4), 1e-4, f32), device, dtype=dt, requires_grad=False)
        with ndl.autocast(dtype=dt):
            loss = ops.summation(x @ W) * 1e-4
        loss.backward()
        res[dt] = W.grad.numpy()[0, 0]
    assert abs(res["float16"] - 3e-8) / 3e-8 > 0.5
    assert abs(res["bfloat16"] - 3e-8) / 3e-8 < 2e-2


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_large_activations_do_not_overflow(device):
    W = T(np.full((4, 2), 300.0, f32), device)
    x = T(np.full((3, 4), 100.0, f32), device, requires_grad=False)  # x @ W = 120000
    with ndl.autocast(dtype="bfloat16"):
        y = x @ W
    assert np.isfinite(y.numpy().astype(f32)).all()
    with ndl.autocast(dtype="float16"):
        assert np.isinf((x @ W).numpy()).all()


def test_grad_scaler_harmless_with_bf16():
    res = []
    for use in (False, True):
        model, (X, y) = mlp(DEV), batch(DEV)
        opt = ndl.optim.SGD(model.parameters(), lr=0.1)
        sc = ndl.amp.GradScaler(init_scale=2.0 ** 10, enabled=use)
        with ndl.autocast(dtype="bfloat16"):
            loss = nn.SoftmaxLoss()(model(X), y)
        sc.scale(loss).backward()
        assert sc.step(opt)
        res.append([p.numpy() for p in model.parameters()])
    for a, b in zip(*res):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- models


@pytest.mark.parametrize("kind", ["rnn", "lstm", "transformer"])
def test_language_models_under_bf16_autocast(kind):
    from models import LanguageModel

    res = {}
    for dt in (None, "float16", "bfloat16"):
        np.random.seed(0)
        model = LanguageModel(16, 30, 32, num_layers=2, seq_model=kind, device=DEV)
        np.random.seed(1)
        x = T(np.random.randint(0, 30, (8, 3)), DEV, requires_grad=False)
        y = T(np.random.randint(0, 30, (24,)), DEV, requires_grad=False)
        with ndl.autocast(enabled=dt is not None, dtype=dt or "float16"):
            loss = nn.SoftmaxLoss()(model(x)[0], y)
        loss.backward()
        assert all(p.dtype == "float32" and p.grad.dtype == "float32" for p in model.parameters())
        res[dt] = (loss.numpy().item(), flat_grads(model))
    l32, g32 = res[None]
    lb, gb = res["bfloat16"]
    assert np.isfinite(gb).all()
    assert abs(lb - l32) < 5e-3 * abs(l32)
    assert rel(gb, g32) < (0.1 if kind == "transformer" else 2e-2)
    # bfloat16 rounds ~8x more coarsely than float16 (3 fewer mantissa bits)
    assert rel(gb, g32) > rel(res["float16"][1], g32)


def test_resnet9_under_bf16_autocast():
    """As for float16, compare with the effect of only rounding weights and input to
    bfloat16 once (all math in float32): ResNet9's freshly initialised BatchNorm
    stack amplifies such perturbations."""
    from models import ResNet9

    def run(amp=False, round_inputs=False):
        np.random.seed(0)
        model = ResNet9(device=DEV)
        if round_inputs:
            for p in model.parameters():
                p.data = T(p.numpy().astype(bf16).astype(f32), DEV, requires_grad=False)
        np.random.seed(1)
        xn = np.random.rand(16, 3, 32, 32).astype(f32)
        if round_inputs:
            xn = xn.astype(bf16).astype(f32)
        x = T(xn, DEV, requires_grad=False)
        y = T(np.random.randint(0, 10, 16), DEV, requires_grad=False)
        with ndl.autocast(enabled=amp, dtype="bfloat16"):
            loss = nn.SoftmaxLoss()(model(x), y)
        loss.backward()
        return loss.numpy().item(), flat_grads(model)

    l32, g32 = run()
    lb, gb = run(amp=True)
    _, g_round = run(round_inputs=True)
    assert np.isfinite(gb).all() and abs(lb - l32) < 5e-3 * abs(l32)
    assert rel(gb, g32) < 2 * rel(g_round, g32) + 1e-3


def test_training_with_bf16_tracks_float32():
    curves = {}
    for amp in (False, True):
        model, (X, y) = mlp(DEV, seed=5), batch(DEV, seed=6)
        opt = ndl.optim.SGD(model.parameters(), lr=0.5)
        losses = []
        for _ in range(15):
            opt.reset_grad()
            with ndl.autocast(enabled=amp, dtype="bfloat16"):
                loss = nn.SoftmaxLoss()(model(X), y)
            loss.backward()
            opt.step()
            losses.append(loss.numpy().item())
        curves[amp] = np.array(losses)
    assert curves[False][-1] < 0.6 * curves[False][0]
    np.testing.assert_allclose(curves[True], curves[False], rtol=5e-2, atol=1e-2)


# --------------------------------------------------------------------------- causal mask


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_bf16_causal_mask(device):
    layer = nn.MultiHeadAttention(causal=True, device=device)
    value = nn.MultiHeadAttention.causal_mask_value("bfloat16")
    assert value == -float(ml_dtypes.finfo(bf16).max) / 2
    m = layer.create_causal_mask(3, 5, device, dtype="bfloat16")
    assert m.dtype == "bfloat16"
    expected = (np.triu(np.ones((1, 1, 3, 5), f32), 3) * f32(value)).astype(bf16)
    np.testing.assert_array_equal(bits(m.numpy()), bits(expected))
    assert np.isfinite(m.numpy().astype(f32)).all()


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_existing_masks_unchanged(dtype):
    """float32 / float16 masks are bit-identical to the previous implementation."""
    layer = nn.MultiHeadAttention(causal=True, device=DEV)
    m = layer.create_causal_mask(4, 6, DEV, dtype=dtype).numpy()
    prev = (-np.finfo(dtype).max / 2) * np.triu(np.ones((1, 1, 4, 6), dtype=dtype), 3)
    np.testing.assert_array_equal(bits(m), bits(prev.astype(dtype)))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_bf16_attention_fully_masked_rows_and_model(device):
    layer = nn.MultiHeadAttention(causal=True, device=device)
    rs = np.random.RandomState(0)
    q = T(rs.randn(2, 2, 8, 8).astype(bf16), device)
    k = T(rs.randn(2, 2, 6, 8).astype(bf16), device)
    out, probs = layer(q, k, k)  # 8 queries, 6 keys: first two rows see no key
    out.sum().backward()
    assert np.isfinite(out.numpy().astype(f32)).all() and np.isfinite(q.grad.numpy().astype(f32)).all()
    model = nn.Transformer(16, 32, 1, num_head=2, dim_head=8, causal=True,
                           device=device, dtype="bfloat16", batch_first=True)
    y, _ = model(T(rs.randn(2, 7, 16).astype(bf16), device))
    y.sum().backward()
    assert y.dtype == "bfloat16" and np.isfinite(y.numpy().astype(f32)).all()


# --------------------------------------------------------------------------- simple_ml amp="bfloat16"


def _ptb():
    from needle.data.datasets.ptb_dataset import batchify

    np.random.seed(0)
    return batchify(list(np.random.randint(0, 40, 1200)), 8, DEV, "float32")


def _lm(kind="lstm"):
    from models import LanguageModel

    np.random.seed(0)
    return LanguageModel(16, 40, 32, num_layers=1, seq_model=kind, device=DEV)


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_simple_ml_ptb_bf16(kind):
    from simple_ml import evaluate_ptb, train_ptb

    lr = 0.05 if kind == "transformer" else 0.5
    res = {}
    for amp in (False, "bfloat16"):
        model = _lm(kind)
        tr = train_ptb(model, _ptb(), seq_len=10, n_epochs=2, lr=lr, clip=0.25, device=DEV, amp=amp)
        ev = evaluate_ptb(model, _ptb(), seq_len=10, device=DEV, amp=amp)
        assert all(p.dtype == "float32" for p in model.parameters())
        res[amp] = (tr[1], ev[1])
    for a, b in zip(res["bfloat16"], res[False]):
        assert abs(a - b) < 2e-2 * b


def test_simple_ml_cifar_bf16():
    from models import ResNet9
    from simple_ml import evaluate_cifar10, train_cifar10

    np.random.seed(0)
    ds = ndl.data.NDArrayDataset(np.random.rand(16, 3, 32, 32).astype(f32), np.random.randint(0, 10, 16))
    np.random.seed(0)
    model = ResNet9(device=DEV)
    tr = train_cifar10(model, ndl.data.DataLoader(ds, 8), n_epochs=1, amp="bfloat16")
    ev = evaluate_cifar10(model, ndl.data.DataLoader(ds, 8), amp="bfloat16")
    assert np.isfinite(tr[1]) and np.isfinite(ev[1])


def test_simple_ml_amp_dtype_and_scaler_choice(monkeypatch):
    import simple_ml

    assert simple_ml._amp_dtype(False) is None and simple_ml._amp_dtype(None) is None
    assert simple_ml._amp_dtype(True) == "float16" and simple_ml._amp_dtype("float16") == "float16"
    assert simple_ml._amp_dtype("bfloat16") == "bfloat16"
    with pytest.raises(ValueError, match="amp must be"):
        simple_ml._amp_dtype("int8")
    opt = object()
    assert simple_ml._make_scaler(True, opt, None).is_enabled()
    assert not simple_ml._make_scaler("bfloat16", opt, None).is_enabled()  # no scaling for bf16

    created = []
    real = ndl.amp.GradScaler

    class Recording(real):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self.is_enabled())

    monkeypatch.setattr(ndl.amp, "GradScaler", Recording)
    simple_ml.train_ptb(_lm(), _ptb(), seq_len=50, n_epochs=1, lr=0.1, device=DEV, amp="bfloat16")
    simple_ml.train_ptb(_lm(), _ptb(), seq_len=50, n_epochs=1, lr=0.1, device=DEV, amp=True)
    assert created == [False, True]


def test_simple_ml_forward_runs_under_bf16_autocast():
    from simple_ml import epoch_general_ptb

    seen = []
    model = _lm("rnn")
    fwd = model.forward

    def spy(*a, **kw):
        seen.append((ndl.is_autocast_enabled(), ndl.get_autocast_dtype()))
        return fwd(*a, **kw)

    model.forward = spy
    epoch_general_ptb(_ptb(), model, seq_len=50, opt=ndl.optim.SGD(model.parameters(), lr=0.1),
                      device=DEV, amp="bfloat16")
    assert seen and all(s == (True, "bfloat16") for s in seen)
