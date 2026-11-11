"""ndl.autocast: automatic mixed precision (mixed precision step 3)."""
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import autograd, ops

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f16, f32 = np.float16, np.float32


def T(x, device, dtype=None, requires_grad=True):
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def bits(x):
    x = np.ascontiguousarray(x)
    return x.view(np.uint16 if x.dtype == f16 else np.uint32)


def cast_nodes(out):
    return [n for n in autograd.find_topo_sort([out]) if isinstance(n.op, ops.Cast)]


class Counting(autograd.TensorOp):
    """y = 2x, counting forward evaluations (to observe recomputation)."""

    calls = 0

    def compute(self, a):
        Counting.calls += 1
        return a * 2.0

    def gradient(self, out_grad, node):
        return out_grad * 2.0


# --------------------------------------------------------------------------- context manager


def test_context_state_and_nesting():
    assert not ndl.is_autocast_enabled()
    with ndl.autocast():
        assert ndl.is_autocast_enabled() and ndl.get_autocast_dtype() == "float16"
        with ndl.autocast(enabled=False):
            assert not ndl.is_autocast_enabled()
        assert ndl.is_autocast_enabled()
    assert not ndl.is_autocast_enabled()
    with pytest.raises(ValueError):
        with ndl.autocast():
            raise ValueError("boom")
    assert not ndl.is_autocast_enabled()  # restored after an exception
    with pytest.raises(TypeError):
        ndl.autocast(dtype="int8")


def test_decorator_and_reentrant_instance():
    ac = ndl.autocast()

    @ac
    def f(depth):
        assert ndl.is_autocast_enabled()
        return f(depth - 1) if depth else ndl.get_autocast_dtype()

    assert f(3) == "float16"
    assert not ndl.is_autocast_enabled()


def test_lazy_mode_ignores_autocast():
    autograd.LAZY_MODE = True
    try:
        a = T(np.ones((2, 2), f32), ndl.cpu())
        with ndl.autocast():
            y = a @ a
        assert y.dtype == "float32"
    finally:
        autograd.LAZY_MODE = False


# --------------------------------------------------------------------------- per-op policies


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_lower_precision_ops(device):
    np.random.seed(0)
    a = T(np.random.randn(4, 5).astype(f32), device)
    b = T(np.random.randn(5, 3).astype(f32), device)
    x = T(np.random.randn(1, 6, 6, 2).astype(f32), device)
    w = T(np.random.randn(3, 3, 2, 4).astype(f32), device)
    assert (a @ b).dtype == "float32"  # outside: unchanged
    with ndl.autocast():
        y = a @ b
        c = ops.conv(x, w, padding=1)
    assert y.dtype == "float16" and c.dtype == "float16"
    # identical to casting by hand
    np.testing.assert_array_equal(bits(y.numpy()), bits((a.half() @ b.half()).numpy()))
    np.testing.assert_array_equal(
        bits(c.numpy()), bits(ops.conv(x.half(), w.half(), padding=1).numpy())
    )


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float32_ops(device):
    np.random.seed(1)
    h = T(np.random.uniform(0.5, 2, (3, 4)).astype(f16), device)
    with ndl.autocast():
        outs = {
            "summation": ops.summation(h, axes=(1,)),
            "exp": ops.exp(h),
            "log": ops.log(h),
            "logsumexp": ops.logsumexp(h, axes=(1,)),
            "logsoftmax": ops.logsoftmax(h),
            "power_scalar": h ** 2,
            "ewise_pow": h ** h,
        }
    for name, y in outs.items():
        assert y.dtype == "float32", name
    # x ** 2 in float32 does not overflow where float16 would
    big = T(np.array([300.0], f16), device)
    with ndl.autocast():
        assert (big ** 2).numpy()[0] == 90000.0
    assert np.isinf((big ** 2).numpy()[0])


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_promote_and_passthrough_ops(device):
    h = T(np.ones((2, 3), f16), device)
    s = T(np.ones((2, 3), f32), device)
    with pytest.raises(TypeError, match="dtype"):  # outside: still an explicit-cast error
        h + s
    with ndl.autocast():
        assert (h + s).dtype == "float32" and (h * s).dtype == "float32"
        assert (h / s).dtype == "float32" and (s - h).dtype == "float32"
        assert (h + h).dtype == "float16" and (h * 3.0 + 1).dtype == "float16"
        assert ops.relu(h).dtype == "float16" and ops.tanh(h).dtype == "float16"
        assert ops.reshape(h, (3, 2)).dtype == "float16"
        assert ops.transpose(h).dtype == "float16"
        assert ops.broadcast_to(ops.reshape(h, (1, 2, 3)), (4, 2, 3)).dtype == "float16"
        assert ops.stack([h, s], 0).dtype == "float32"
        assert ops.split(h, 0)[0].dtype == "float16"
    with pytest.raises(TypeError, match="dtype"):
        ops.stack([h, s], 0)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_linear_output_dtype(device):
    """float16 matmul + float32 bias -> float32; without bias the result stays float16."""
    x = T(np.ones((2, 4), f32), device)
    with ndl.autocast():
        assert nn.Linear(4, 3, device=device)(x).dtype == "float32"
        assert nn.Linear(4, 3, bias=False, device=device)(x).dtype == "float16"


# --------------------------------------------------------------------------- gradients


def _mlp(device, seed=0):
    np.random.seed(seed)
    X = T(np.random.randn(16, 8).astype(f32), device, requires_grad=False)
    y = T(np.random.randint(0, 4, 16), device, requires_grad=False)
    model = nn.Sequential(nn.Linear(8, 32, device=device), nn.ReLU(), nn.Linear(32, 4, device=device))
    return model, X, y


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_autocast_equals_manual_casts(device):
    """Autocast is exactly 'insert these casts': same values and gradients bit for bit."""
    grads = {}
    for mode in ("auto", "manual"):
        model, X, y = _mlp(device)
        l1, l2 = model.modules[0], model.modules[2]
        if mode == "auto":
            with ndl.autocast():
                loss = nn.SoftmaxLoss()(model(X), y)
        else:
            h = X.half() @ l1.weight.half()
            h = h.float() + ops.broadcast_to(ops.reshape(l1.bias, (1, 32)), (16, 32))
            h = ops.relu(h)
            o = (h.half() @ l2.weight.half()).float() + ops.broadcast_to(
                ops.reshape(l2.bias, (1, 4)), (16, 4)
            )
            loss = nn.SoftmaxLoss()(o, y)
        loss.backward()
        grads[mode] = [loss.numpy()] + [p.grad.numpy() for p in model.parameters()]
        assert all(p.grad.dtype == "float32" for p in model.parameters())
    for a, b in zip(grads["auto"], grads["manual"]):
        np.testing.assert_array_equal(bits(a), bits(b))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_backward_inside_or_outside_region(device):
    res = []
    for inside in (False, True):
        model, X, y = _mlp(device)
        with ndl.autocast():
            loss = nn.SoftmaxLoss()(model(X), y)
            if inside:
                loss.backward()
        if not inside:
            loss.backward()
        res.append([p.grad.numpy() for p in model.parameters()])
    for a, b in zip(*res):
        np.testing.assert_array_equal(bits(a), bits(b))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_backward_of_float32_graph_unaffected_by_region(device):
    """Calling backward inside autocast must not lower a float32 graph's gradients."""
    res = []
    for inside in (False, True):
        model, X, y = _mlp(device)
        loss = nn.SoftmaxLoss()(model(X), y)  # float32 forward
        with ndl.autocast(enabled=inside):
            loss.backward()
        res.append([p.grad.numpy() for p in model.parameters()])
    for a, b in zip(*res):
        assert a.dtype == f32
        np.testing.assert_array_equal(bits(a), bits(b))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_gradients_close_to_float32(device):
    res = {}
    for amp in (False, True):
        model, X, y = _mlp(device, seed=3)
        with ndl.autocast(enabled=amp):
            loss = nn.SoftmaxLoss()(model(X), y)
        loss.backward()
        res[amp] = (loss.numpy(), np.concatenate([p.grad.numpy().ravel() for p in model.parameters()]))
    np.testing.assert_allclose(res[True][0], res[False][0], rtol=1e-3)
    g32, g16 = res[False][1], res[True][1]
    assert np.linalg.norm(g16 - g32) / np.linalg.norm(g32) < 5e-3


# --------------------------------------------------------------------------- parameter cast cache


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_parameter_cast_is_cached_per_region(device):
    W = T(np.ones((3, 3), f32), device)
    x = T(np.ones((2, 3), f32), device, requires_grad=False)
    with ndl.autocast():
        out = ops.summation(x @ W + x @ W)
        assert len([n for n in cast_nodes(out) if n.inputs[0] is W]) == 1
        cached = [n for n in cast_nodes(out) if n.inputs[0] is W][0]
    with ndl.autocast():  # a new region casts again
        out2 = ops.summation(x @ W)
        assert [n for n in cast_nodes(out2) if n.inputs[0] is W][0] is not cached
    with ndl.autocast(cache_enabled=False):
        out3 = ops.summation(x @ W + x @ W)
        assert len([n for n in cast_nodes(out3) if n.inputs[0] is W]) == 2
    out.backward()
    np.testing.assert_array_equal(W.grad.numpy(), np.full((3, 3), 4.0, f32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cache_invalidated_when_parameter_data_changes(device):
    W = T(np.ones((2, 2), f32), device)
    x = T(np.ones((1, 2), f32), device, requires_grad=False)
    with ndl.autocast():
        assert (x @ W).numpy().tolist() == [[2.0, 2.0]]
        W.data = T(np.full((2, 2), 3.0, f32), device)  # e.g. an optimizer step
        assert (x @ W).numpy().tolist() == [[6.0, 6.0]]


def test_rnn_weights_cast_once():
    np.random.seed(0)
    dev = ndl.cpu()
    rnn = nn.RNN(4, 8, num_layers=1, device=dev)
    X = T(np.random.randn(10, 2, 4).astype(f32), dev, requires_grad=False)
    with ndl.autocast():
        out, _ = rnn(X)
    W_hh = rnn.rnn_cells[0].W_hh
    assert len([n for n in cast_nodes(ops.summation(out)) if n.inputs[0] is W_hh]) == 1


def test_float16_adjoints_accumulate_in_float32():
    """A float16 node used 2100 times: a float16 running sum of its 2100 unit
    adjoints would stop at 2048 (2048 + 1 rounds back to 2048)."""
    W = T(np.ones((1,), f32), ndl.cpu())
    h = W.half()
    # a flat graph (stack) rather than a 2100-deep chain of additions
    total = ops.summation(ops.stack([h * 1.0 for _ in range(2100)], 0))
    total.backward()
    assert h.grad.dtype == "float16" and h.grad.numpy()[0] == 2100.0
    assert W.grad.numpy()[0] == 2100.0


# --------------------------------------------------------------------------- casts and recomputation


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_autocast_casts_do_not_release_by_default(device):
    """An activation feeding a float16 matmul and reused later is not recomputed."""
    x = T(np.ones((2, 3), f32), device)
    W = T(np.ones((3, 3), f32), device)
    for release, expected_calls in ((False, 1), (True, 2)):
        Counting.calls = 0
        with ndl.autocast(release_inputs=release):
            a = Counting()(x)  # float32 activation
            y = a @ W  # autocast casts `a` to float16
            z = ops.summation(a) + ops.summation(y)  # `a` used again
        assert Counting.calls == expected_calls
        np.testing.assert_array_equal(z.numpy(), np.array(12.0 + 36.0, f32))


# --------------------------------------------------------------------------- whole models


def _flat_grads(model):
    return np.concatenate([p.grad.numpy().astype(np.float64).ravel() for p in model.parameters()])


@pytest.mark.parametrize("kind", ["rnn", "lstm", "transformer"])
def test_language_models_under_autocast(kind):
    from models import LanguageModel

    dev = ndl.cpu()
    res = {}
    for amp in (False, True):
        np.random.seed(0)
        model = LanguageModel(16, 30, 32, num_layers=2, seq_model=kind, device=dev)
        np.random.seed(1)
        x = T(np.random.randint(0, 30, (8, 3)), dev, requires_grad=False)
        y = T(np.random.randint(0, 30, (24,)), dev, requires_grad=False)
        with ndl.autocast(enabled=amp):
            out, _ = model(x)
            loss = nn.SoftmaxLoss()(out, y)
        loss.backward()
        assert all(p.dtype == "float32" and p.grad.dtype == "float32" for p in model.parameters())
        res[amp] = (loss.numpy().item(), _flat_grads(model))
    assert loss.dtype == "float32"
    assert abs(res[True][0] - res[False][0]) < 1e-3 * abs(res[False][0])
    g32, g16 = res[False][1], res[True][1]
    assert np.isfinite(g16).all()
    assert np.linalg.norm(g16 - g32) / np.linalg.norm(g32) < 2e-2


def test_resnet9_under_autocast():
    """ResNet9's freshly initialised BatchNorm stack is very sensitive to float16 sized
    perturbations: rounding only its weights and input to float16 once (all math in
    float32) already moves the gradients by several percent. Autocast must stay in
    that range, not exceed it by much."""
    from models import ResNet9

    dev = ndl.cpu()

    def run(amp=False, round_inputs=False):
        np.random.seed(0)
        model = ResNet9(device=dev)
        if round_inputs:
            for p in model.parameters():
                p.data = T(p.numpy().astype(f16).astype(f32), dev, requires_grad=False)
        np.random.seed(1)
        xn = np.random.rand(16, 3, 32, 32).astype(f32)
        if round_inputs:
            xn = xn.astype(f16).astype(f32)
        x = T(xn, dev, requires_grad=False)
        y = T(np.random.randint(0, 10, 16), dev, requires_grad=False)
        with ndl.autocast(enabled=amp):
            loss = nn.SoftmaxLoss()(model(x), y)
        loss.backward()
        assert all(p.grad.dtype == "float32" for p in model.parameters())
        return loss.numpy().item(), _flat_grads(model)

    l32, g32 = run()
    l16, g16 = run(amp=True)
    _, g_round = run(round_inputs=True)
    assert np.isfinite(g16).all() and abs(l16 - l32) < 1e-3 * abs(l32)
    err = lambda g: np.linalg.norm(g - g32) / np.linalg.norm(g32)
    assert err(g16) < 2 * err(g_round) + 1e-3


def test_training_with_autocast_tracks_float32():
    """A few SGD steps with autocast follow the float32 loss curve closely."""
    dev = ndl.cpu()
    curves = {}
    for amp in (False, True):
        model, X, y = _mlp(dev, seed=5)
        opt = ndl.optim.SGD(model.parameters(), lr=0.5)
        losses = []
        for _ in range(15):
            opt.reset_grad()
            with ndl.autocast(enabled=amp):
                loss = nn.SoftmaxLoss()(model(X), y)
            loss.backward()
            opt.step()
            losses.append(loss.numpy().item())
        assert all(p.dtype == "float32" for p in model.parameters())
        curves[amp] = np.array(losses)
    assert curves[False][-1] < 0.5 * curves[False][0]  # it actually trains
    np.testing.assert_allclose(curves[True], curves[False], rtol=2e-2, atol=2e-3)
