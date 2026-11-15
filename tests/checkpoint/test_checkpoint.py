"""needle.checkpoint: the same gradients as without checkpoints, with the activations
of each segment released between forward and backward.
Design: agent-docs/gradient-checkpointing/ (decisions 01-09)."""
import gc
import sys
import weakref

sys.path.append("./python")
import numpy as np
import pytest

import needle as ndl
from needle import autograd, nn, ops

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]


def T(shape, device, requires_grad=True, scale=1.0):
    return ndl.Tensor((np.random.randn(*shape) * scale).astype(np.float32), device=device,
                      requires_grad=requires_grad)


class Counting(autograd.TensorOp):
    """y = 2 * x that counts how often its forward runs."""

    calls = 0

    def compute(self, a):
        Counting.calls += 1
        return a * 2.0

    def gradient(self, out_grad, node):
        return out_grad * 2.0


def total(out):
    """Sum of every tensor in a nested result, as one scalar loss."""
    if isinstance(out, (tuple, list)):
        parts = [total(o) for o in out]
        return parts[0] if len(parts) == 1 else sum(parts[1:], parts[0])
    return ops.summation(out.float() if out.dtype != "float32" else out)


# ------------------------------------------------------------------ models used as segments


def mlp(device):
    W1, W2 = T((16, 32), device, scale=0.3), T((32, 16), device, scale=0.3)
    return (lambda h: ops.relu(ops.relu(h @ W1) @ W2)), T((8, 16), device, requires_grad=False), [W1, W2], 2


def transformer(device):
    layer = nn.TransformerLayer(16, 2, 8, 32, device=device)
    return layer, T((2, 8, 16), device, requires_grad=False), layer.parameters(), 2


def lstm(device):
    model = nn.LSTM(8, 16, num_layers=2, device=device)
    return model, T((5, 3, 8), device, requires_grad=False), model.parameters(), 1


MODELS = {"mlp": mlp, "transformer": transformer, "lstm": lstm}


def train_step(make, device, use_checkpoint, amp=None):
    """Gradients of one step; `segments` chained segments (LSTM: one, nested result)."""
    np.random.seed(0)
    fn, x, params, segments = make(device)
    out = x
    with ndl.autocast(dtype=amp) if amp else ndl.autocast(enabled=False):
        for _ in range(segments):
            out = ndl.checkpoint(fn, out) if use_checkpoint else fn(out)
        loss = total(out)
    loss.backward()
    return [p.grad.numpy() for p in params]


def assert_same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


# ------------------------------------------------------------------ 1-4: same results


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_gradients_bit_identical(model, device):
    assert_same(train_step(MODELS[model], device, False), train_step(MODELS[model], device, True))


@pytest.mark.parametrize("amp", ["float16", "bfloat16"])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_gradients_bit_identical_under_autocast(amp, model, device):
    assert_same(train_step(MODELS[model], device, False, amp), train_step(MODELS[model], device, True, amp))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_dropout_mask_is_reused(device):
    def step(use_checkpoint):
        np.random.seed(0)
        W = T((16, 16), device, scale=0.3)
        drop = nn.Dropout(0.5)
        fn = lambda h: drop(ops.relu(h @ W))
        x = T((8, 16), device, requires_grad=False)
        out = ndl.checkpoint(fn, x) if use_checkpoint else fn(x)
        ops.summation(out * out).backward()
        return [W.grad.numpy()]

    assert_same(step(False), step(True))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_batchnorm_statistics_updated_once(device):
    def step(use_checkpoint):
        np.random.seed(0)
        W = T((16, 16), device, scale=0.3)
        bn = nn.BatchNorm1d(16, device=device)
        fn = lambda h: ops.relu(bn(h @ W))
        x = T((8, 16), device, requires_grad=False)
        out = ndl.checkpoint(fn, x) if use_checkpoint else fn(x)
        ops.summation(out).backward()
        return [bn.running_mean.numpy(), bn.running_var.numpy(), W.grad.numpy()]

    assert_same(step(False), step(True))


# ------------------------------------------------------------------ 5-9: what is released


def segment_with_probes(device, inner):
    """fn(h) = relu(Counting(h @ W)); appends its interior tensors to `inner`."""
    W = T((16, 16), device, scale=0.3)

    def fn(h):
        a = h @ W
        b = Counting()(a)
        inner.extend([a, b])
        return ops.relu(b)

    return fn, W


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_interior_released_after_forward(device):
    inner = []
    fn, _ = segment_with_probes(device, inner)
    x = T((8, 16), device)
    refs = []

    def segment(h):
        result = fn(h)
        refs.extend(weakref.ref(t.cached_data) for t in inner)  # before checkpoint() releases them
        return result

    out = ndl.checkpoint(segment, x)
    gc.collect()
    assert all(r() is None for r in refs), "interior buffers should be freed"
    assert all(t.cached_data is None and t._checkpoint_interior for t in inner)
    assert out.cached_data is not None and not out._checkpoint_interior
    assert x.cached_data is not None


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_interior_released_after_backward(device):
    inner = []
    fn, W = segment_with_probes(device, inner)
    out = ndl.checkpoint(fn, T((8, 16), device))
    loss = ops.summation(out)
    loss.backward()
    # recomputed during backward, then released again; their adjoints are dropped too
    assert all(t.cached_data is None for t in inner)
    assert all(t.grad is None for t in inner)
    assert out.cached_data is not None and out.grad is not None
    assert W.grad is not None


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_recompute_count(device):
    for use_checkpoint, expected in ((False, 1), (True, 2)):
        Counting.calls = 0
        fn, _ = segment_with_probes(device, [])
        x = T((8, 16), device)
        out = ndl.checkpoint(fn, x) if use_checkpoint else fn(x)
        assert Counting.calls == 1
        ops.summation(out).backward()  # relu's gradient reads Counting's output
        assert Counting.calls == expected


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_lstm_tensortuples_released(device):
    np.random.seed(0)
    model, x, _, _ = lstm(device)
    out, (h, c) = ndl.checkpoint(model, x)
    tuples = [n for n in autograd.find_topo_sort([out, h, c])
              if isinstance(n, autograd.TensorTuple) and not n.is_leaf()]
    assert {type(n.op).__name__ for n in tuples} == {"Split", "MakeTensorTuple"}
    assert all(n.cached_data is None for n in tuples)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_nested_outputs_kept(device):
    np.random.seed(0)
    model, x, _, _ = lstm(device)
    out, (h, c) = ndl.checkpoint(model, x)
    assert all(t.cached_data is not None for t in (out, h, c))
    W = T((4, 4), device)
    a, [b, (d,)] = ndl.checkpoint(lambda v: (v @ W, [ops.relu(v @ W), (v @ W * 2.0,)]), T((3, 4), device))
    assert all(t.cached_data is not None for t in (a, b, d))


# ------------------------------------------------------------------ 10-14: modes and limits


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_no_grad_is_plain_call(device):
    inner = []
    fn, _ = segment_with_probes(device, inner)
    x = T((8, 16), device)
    with ndl.no_grad():
        out = ndl.checkpoint(fn, x)
        ref = fn(x)
    np.testing.assert_array_equal(out.numpy(), ref.numpy())
    assert out.is_leaf() and all(t.is_leaf() and t.cached_data is not None for t in inner)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_interior_read_after_step_uses_new_weights(device):
    """The documented limit: a released activation is recomputed from current parameters."""
    inner = []
    fn, W = segment_with_probes(device, inner)
    x = T((8, 16), device, requires_grad=False)
    out = ndl.checkpoint(fn, x)
    a_forward, out_forward = (x @ W).numpy(), out.numpy()
    ops.summation(out).backward()
    ndl.optim.SGD([W], lr=0.5).step()
    np.testing.assert_array_equal(inner[0].numpy(), (x @ W).numpy())
    assert not np.array_equal(inner[0].numpy(), a_forward)
    np.testing.assert_array_equal(out.numpy(), out_forward)  # the segment output keeps its value


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_second_backward_same_gradients(device):
    np.random.seed(0)
    fn, x, params, segments = mlp(device)
    out = x
    for _ in range(segments):
        out = ndl.checkpoint(fn, out)
    loss = ops.summation(out)
    loss.backward()
    first = [p.grad.numpy() for p in params]
    loss.backward()
    assert_same(first, [p.grad.numpy() for p in params])


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_releases_segment_input(device):
    """Decision 07: an explicit cast in a segment releases the segment input (the previous
    segment's output); backward recomputes it and the gradients stay identical."""
    def step(use_checkpoint):
        np.random.seed(0)
        W1, W2 = T((16, 16), device, scale=0.3), T((16, 16), device, scale=0.3)
        seg1 = lambda h: ops.relu(h @ W1)
        seg2 = lambda h: ops.relu(h.half() @ W2.half()).float()
        x = T((8, 16), device, requires_grad=False)
        if use_checkpoint:
            h = ndl.checkpoint(seg1, x)
            out = ndl.checkpoint(seg2, h)
            assert h.cached_data is None  # released by the cast in seg2
        else:
            out = seg2(seg1(x))
        ops.summation(out).backward()
        return [W1.grad.numpy(), W2.grad.numpy()]

    assert_same(step(False), step(True))
