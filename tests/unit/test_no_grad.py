"""ndl.no_grad / enable_grad / set_grad_enabled and graph-free backward
(backward(create_graph=False) by default): what is recorded, and what memory
stays alive."""
import gc
import sys
import weakref

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
f32 = np.float32


def T(x, device, requires_grad=True, dtype=None):
    return ndl.Tensor(x, device=device, requires_grad=requires_grad, dtype=dtype)


def mlp(device, seed=0):
    np.random.seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32, device=device), nn.ReLU(), nn.Linear(32, 32, device=device),
        nn.ReLU(), nn.Linear(32, 4, device=device),
    )


def batch(device, seed=1, n=8):
    np.random.seed(seed)
    X = T(np.random.randn(n, 16).astype(f32), device, requires_grad=False)
    y = T(np.random.randint(0, 4, n), device, requires_grad=False)
    return X, y


def activation_refs(out, model):
    """Weak references to every op-produced tensor's data in out's forward graph."""
    params = {id(p) for p in model.parameters()}
    return [
        (weakref.ref(n.cached_data), int(np.prod(n.shape)))
        for n in autograd.find_topo_sort([out])
        if isinstance(n, ndl.Tensor)  # TensorTuple data is a plain tuple
        and n.op is not None and id(n) not in params and n.cached_data is not None
    ]


def alive_fraction(refs):
    gc.collect()
    return sum(s for r, s in refs if r() is not None) / sum(s for _, s in refs)


# --------------------------------------------------------------------------- grad mode


def test_grad_mode_contexts():
    assert ndl.is_grad_enabled()
    with ndl.no_grad():
        assert not ndl.is_grad_enabled()
        with ndl.enable_grad():
            assert ndl.is_grad_enabled()
        assert not ndl.is_grad_enabled()
        with ndl.set_grad_enabled(True):
            assert ndl.is_grad_enabled()
    assert ndl.is_grad_enabled()
    with ndl.set_grad_enabled(False):
        assert not ndl.is_grad_enabled()
    with pytest.raises(ValueError):
        with ndl.no_grad():
            raise ValueError("boom")
    assert ndl.is_grad_enabled()  # restored after an exception


def test_decorator_and_reentrant_instance():
    ng = ndl.no_grad()

    @ng
    def f(depth):
        assert not ndl.is_grad_enabled()
        return f(depth - 1) if depth else "done"

    assert f(3) == "done" and ndl.is_grad_enabled()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_ops_under_no_grad_are_detached(device):
    w = T(np.random.randn(4, 3).astype(f32), device)
    x = T(np.random.randn(2, 4).astype(f32), device)
    y_grad = ops.relu(x @ w) * 2.0 + 1.0
    with ndl.no_grad():
        y = ops.relu(x @ w) * 2.0 + 1.0
        parts = ops.split(ops.stack([x, x], 0), 0)  # TensorTuple ops too
        leaf = T(np.ones(3, f32), device)  # explicit leaves keep requires_grad
        p = nn.Parameter(np.ones(3, f32), device=device)
    assert not y.requires_grad and y.is_leaf() and y.inputs == []
    np.testing.assert_array_equal(y.numpy(), y_grad.numpy())  # same values
    assert not parts[0].requires_grad
    assert leaf.requires_grad and p.requires_grad
    assert y_grad.requires_grad and not y_grad.is_leaf()  # outside: recorded as before


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_no_grad_frees_activations_during_forward(device):
    model = mlp(device)
    X, _ = batch(device)
    for grad in (True, False):
        with ndl.set_grad_enabled(grad):
            h = model.modules[0](X)
            ref = weakref.ref(h.cached_data)
            out = model.modules[2](model.modules[1](h))
            del h
            gc.collect()
        assert (ref() is not None) == grad  # kept only when a graph references it
        del out


def test_backward_on_tensor_without_grad_raises():
    model = mlp(ndl.cpu())
    X, y = batch(ndl.cpu())
    with ndl.no_grad():
        loss = nn.SoftmaxLoss()(model(X), y)
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()
    with pytest.raises(RuntimeError, match="does not require grad"):
        T(np.ones(2, f32), ndl.cpu(), requires_grad=False).sum().backward()


# --------------------------------------------------------------------------- backward without a graph


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_grads_are_plain_tensors_and_identical(device):
    res = {}
    for create_graph in (False, True):
        model = mlp(device)
        X, y = batch(device)
        nn.SoftmaxLoss()(model(X), y).backward(create_graph=create_graph)
        for p in model.parameters():
            assert p.grad.is_leaf() != create_graph
            assert p.grad.requires_grad == create_graph
        res[create_graph] = [p.grad.numpy() for p in model.parameters()]
    for a, b in zip(res[False], res[True]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_forward_activations_freed_after_backward(device):
    for create_graph, expect_alive in ((False, 0.0), (True, 0.9)):
        model = mlp(device)
        X, y = batch(device)
        loss = nn.SoftmaxLoss()(model(X), y)
        refs = activation_refs(loss, model)
        loss.backward(create_graph=create_graph)
        del loss
        frac = alive_fraction(refs)
        if create_graph:
            assert frac > expect_alive  # the .grad graphs hold the activations
        else:
            assert frac == 0.0


def test_grad_mode_restored_by_backward():
    model = mlp(ndl.cpu())
    X, y = batch(ndl.cpu())
    loss = nn.SoftmaxLoss()(model(X), y)
    with ndl.no_grad():  # backward inside no_grad of a graph built outside
        loss.backward(create_graph=True)
        assert not ndl.is_grad_enabled()
    assert not model.modules[0].weight.grad.is_leaf()  # create_graph wins inside backward
    loss2 = nn.SoftmaxLoss()(model(X), y)
    loss2.backward()
    assert ndl.is_grad_enabled()


def test_second_order_gradients_with_create_graph():
    x = T(np.array([1.0, 2.0, 3.0], f32), ndl.cpu())
    ops.summation(x ** 3).backward(create_graph=True)
    g = x.grad
    np.testing.assert_allclose(g.numpy(), 3 * np.array([1, 4, 9]))
    x.grad = None
    ops.summation(g).backward()
    np.testing.assert_allclose(x.grad.numpy(), [6.0, 12.0, 18.0])
    # Hessian-vector product of tanh: d/dx sum(v * d tanh(x)/dx)
    x = T(np.array([0.1, -0.5, 1.0], f32), ndl.cpu())
    v = np.array([1.0, 2.0, -1.0], f32)
    ops.summation(ops.tanh(x)).backward(create_graph=True)
    g = x.grad
    x.grad = None
    ops.summation(g * T(v, ndl.cpu(), requires_grad=False)).backward()
    t = np.tanh(x.numpy())
    np.testing.assert_allclose(x.grad.numpy(), v * (-2 * t * (1 - t ** 2)), rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_models_identical_with_and_without_graph(device):
    """Graph-free backward changes no gradient: LSTM, Transformer, conv."""
    from models import LanguageModel

    def lm_grads(kind, create_graph):
        np.random.seed(0)
        m = LanguageModel(8, 20, 16, num_layers=1, seq_model=kind, device=device)
        np.random.seed(1)
        x = T(np.random.randint(0, 20, (6, 2)), device, requires_grad=False)
        out, _ = m(x)
        ops.summation(out).backward(create_graph=create_graph)
        return [p.grad.numpy() for p in m.parameters()]

    for kind in ("lstm", "transformer"):
        for a, b in zip(lm_grads(kind, False), lm_grads(kind, True)):
            np.testing.assert_array_equal(a, b)

    def conv_grads(create_graph):
        np.random.seed(2)
        conv = nn.Conv(3, 4, 3, stride=2, device=device)
        x = T(np.random.randn(2, 3, 7, 7).astype(f32), device)
        ops.summation(conv(x) ** 2).backward(create_graph=create_graph)
        return [p.grad.numpy() for p in conv.parameters()] + [x.grad.numpy()]

    for a, b in zip(conv_grads(False), conv_grads(True)):
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- interplay


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_autocast_cast_cache_respects_grad_mode(device):
    """A parameter cast under no_grad (a detached copy) is not reused for training
    in the same autocast region, so the parameter still gets its gradient."""
    model = mlp(device)
    X, y = batch(device)
    with ndl.autocast():
        with ndl.no_grad():
            model(X)  # e.g. a quick evaluation inside the region
        loss = nn.SoftmaxLoss()(model(X), y)
    loss.backward()
    assert all(p.grad is not None and p.grad.dtype == "float32" for p in model.parameters())
    ref = mlp(device)
    with ndl.autocast():
        nn.SoftmaxLoss()(ref(X), y).backward()
    for a, b in zip(model.parameters(), ref.parameters()):
        np.testing.assert_array_equal(a.grad.numpy(), b.grad.numpy())


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_optimizers_and_scaler_with_plain_grads(device):
    for opt_cls in (ndl.optim.SGD, ndl.optim.Adam):
        model = mlp(device)
        X, y = batch(device)
        opt = opt_cls(model.parameters(), lr=0.01)
        sc = ndl.amp.GradScaler(init_scale=2.0 ** 8)
        before = [p.numpy().copy() for p in model.parameters()]
        with ndl.autocast():
            loss = nn.SoftmaxLoss()(model(X), y)
        sc.scale(loss).backward()
        assert sc.step(opt)
        sc.update()
        assert all(not np.array_equal(p.numpy(), b) for p, b in zip(model.parameters(), before))
        assert all(p.is_leaf() and p.dtype == "float32" for p in model.parameters())


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_released_node_recomputed_under_graph_free_backward(device):
    """release_cached_data + backward: the recompute still works without a graph."""
    x = T(np.array([0.2, -0.4, 0.9], f32), device)
    h = ops.tanh(x * 2.0)
    out = ops.summation(h * 3.0)
    h.release_cached_data()
    out.backward()
    t = np.tanh(np.array([0.2, -0.4, 0.9], f32) * 2)
    np.testing.assert_allclose(x.grad.numpy(), 6 * (1 - t ** 2), rtol=1e-5)


def test_lazy_mode_with_no_grad():
    autograd.LAZY_MODE = True
    try:
        x = T(np.ones(3, f32), ndl.cpu())
        with ndl.no_grad():
            y = x * 2.0
        assert not y.requires_grad
        np.testing.assert_array_equal(y.numpy(), np.full(3, 2.0, f32))
    finally:
        autograd.LAZY_MODE = False


# --------------------------------------------------------------------------- training loops


class _Spy(nn.Module):
    """Wraps a model; at each forward records grad mode and how much of the
    previous batch's forward graph is still alive."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.prev = []
        self.alive = []
        self.grad_modes = []

    def forward(self, x, *rest):
        if self.prev:
            self.alive.append(alive_fraction(self.prev))
        self.grad_modes.append(ndl.is_grad_enabled())
        out = self.inner(x, *rest)
        first = out[0] if isinstance(out, tuple) else out
        self.prev = activation_refs(first, self.inner) if first.requires_grad else []
        return out


def test_cifar_epoch_frees_previous_batch_and_eval_has_no_graph():
    from simple_ml import epoch_general_cifar10

    dev = ndl.cpu()
    spy = _Spy(mlp(dev))
    np.random.seed(0)
    ds = ndl.data.NDArrayDataset(np.random.randn(64, 16).astype(f32), np.random.randint(0, 4, 64))
    epoch_general_cifar10(ndl.data.DataLoader(ds, 16), spy, nn.SoftmaxLoss(),
                          ndl.optim.SGD(spy.inner.parameters(), lr=0.1))
    assert spy.alive and max(spy.alive) == 0.0  # previous batch freed before next forward
    assert all(spy.grad_modes)
    spy.alive, spy.grad_modes, spy.prev = [], [], []
    epoch_general_cifar10(ndl.data.DataLoader(ds, 16), spy, nn.SoftmaxLoss(), None)
    assert spy.grad_modes and not any(spy.grad_modes)  # evaluation records nothing
    assert spy.alive == []


def test_ptb_epoch_frees_previous_batch_and_eval_has_no_graph():
    from models import LanguageModel
    from needle.data.datasets.ptb_dataset import batchify
    from simple_ml import epoch_general_ptb

    dev = ndl.cpu()
    np.random.seed(0)
    data = batchify(list(np.random.randint(0, 30, 600)), 4, dev, "float32")
    np.random.seed(0)
    spy = _Spy(LanguageModel(8, 30, 16, num_layers=1, seq_model="lstm", device=dev))
    epoch_general_ptb(data, spy, seq_len=20, opt=ndl.optim.SGD(spy.inner.parameters(), lr=0.1),
                      device=dev)
    assert spy.alive and max(spy.alive) == 0.0 and all(spy.grad_modes)
    spy.alive, spy.grad_modes, spy.prev = [], [], []
    epoch_general_ptb(data, spy, seq_len=20, device=dev)
    assert spy.grad_modes and not any(spy.grad_modes)


def test_evaluation_results_unchanged_by_no_grad():
    """Evaluating under no_grad gives exactly the same accuracy and loss."""
    from simple_ml import evaluate_cifar10

    dev = ndl.cpu()
    np.random.seed(0)
    ds = ndl.data.NDArrayDataset(np.random.randn(32, 16).astype(f32), np.random.randint(0, 4, 32))
    model = mlp(dev)
    acc, loss = evaluate_cifar10(model, ndl.data.DataLoader(ds, 8))
    model.eval()
    X = T(ds.arrays[0], dev, requires_grad=False)
    y = ds.arrays[1]
    logits = model(X)
    ref_loss = nn.SoftmaxLoss()(logits, T(y, dev, requires_grad=False)).numpy().item()
    assert acc == np.mean(np.argmax(logits.numpy(), 1) == y)
    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
