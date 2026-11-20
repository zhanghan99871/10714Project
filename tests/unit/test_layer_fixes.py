"""Regression tests for the bug_fix_layer branch:

* nn.Dropout with p=0 (and in eval mode) is the identity: it no longer creates an
  all-ones mask that the graph keeps until backward ends; with p>0 the graph keeps
  one tensor of the input's size (the product) next to the scaled mask, not two.
* nn.Linear under autocast adds its bias in the matmul's half precision (the bias
  parameter is cast once per region, before it is broadcast), so the output stays
  float16 / bfloat16 instead of being promoted to float32.
"""
import gc
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import autograd, init
from needle.backend_ndarray import ndarray as nd
from needle.policy import measure_step

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f32 = np.float32


def T(shape, device, dtype="float32", requires_grad=True, seed=0):
    x = np.random.RandomState(seed).randn(*shape).astype(f32)
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


# --------------------------------------------------------------------------- Dropout


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_dropout_zero_and_eval_are_the_identity(device):
    x = T((32, 64), device)
    gc.collect()
    before = nd.memory_allocated(device)
    assert nn.Dropout(0.0)(x) is x  # training mode, p = 0
    drop = nn.Dropout(0.5)
    drop.eval()
    assert drop(x) is x
    assert nd.memory_allocated(device) == before  # no mask was made


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_dropout_values_and_gradient(device):
    p = 0.3
    x = T((16, 32), device)
    np.random.seed(5)
    out = nn.Dropout(p)(x)
    np.random.seed(5)  # the same random draws as the layer: the same mask
    mask = init.randb(16, 32, p=1 - p, device=device, dtype="float32", requires_grad=False).numpy()
    assert 0 < mask.mean() < 1
    np.testing.assert_allclose(out.numpy(), x.numpy() * mask / (1 - p), rtol=1e-6, atol=1e-7)
    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), mask / (1 - p), rtol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_dropout_keeps_one_tensor_besides_the_mask(device):
    x = T((64, 64), device)
    nbytes = 64 * 64 * 4
    gc.collect()
    before = nd.memory_allocated(device)
    out = nn.Dropout(0.5)(x)
    gc.collect()
    assert nd.memory_allocated(device) - before == 2 * nbytes  # scaled mask + output
    del out


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_dropout_half(device, dtype):
    p = 0.25
    x = T((8, 16), device, dtype=dtype)
    np.random.seed(1)
    out = nn.Dropout(p)(x)
    assert out.dtype == dtype
    o, xv = out.numpy().astype(f32), x.numpy().astype(f32)
    kept = o != 0
    assert 0 < kept.mean() < 1
    np.testing.assert_allclose(o[kept], xv[kept] / (1 - p), rtol=1e-2)


def test_transformer_dropout_zero_costs_no_memory():
    """A transformer with dropout=0 now uses the same memory in training mode as in
    eval mode; with dropout > 0 the masks cost memory again."""
    from models import LanguageModel

    dev = ndl.cpu()
    rng = np.random.RandomState(1)
    x = ndl.Tensor(rng.randint(0, 50, (24, 4)), device=dev, requires_grad=False)
    y = ndl.Tensor(rng.randint(0, 50, 96), device=dev, requires_grad=False)
    peaks = {}
    for dropout in (0.0, 0.1):
        np.random.seed(0)
        m = LanguageModel(32, 50, 64, num_layers=2, seq_model="transformer", device=dev, policies="fp32")
        for layer in m.sequence_model.layers.modules:
            layer.dropout.p = dropout
            layer.attn_layer.attn.dropout.p = dropout
        loss_fn = lambda: nn.SoftmaxLoss()(m(x)[0], y)
        for mode in ("train", "eval"):
            getattr(m, mode)()
            peaks[dropout, mode] = measure_step(m.sequence_model, loss_fn, "fp32", params=m.parameters(),
                                                warmup=0, repeats=1)["peak"]
    assert peaks[0.0, "train"] == peaks[0.0, "eval"] == peaks[0.1, "eval"]
    assert peaks[0.1, "train"] > peaks[0.0, "train"]


# --------------------------------------------------------------------------- Linear


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_linear_bias_added_in_half_precision(device, dtype):
    lin = nn.Linear(4, 3, device=device)
    x = T((5, 4), device, requires_grad=False)
    with ndl.autocast(dtype=dtype):
        a = lin(x)
        b = lin(x)
        # the bias parameter itself is cast (once per region), not its broadcast
        key = (id(lin.bias), dtype, True)
        assert key in autograd._AUTOCAST.cache
        assert autograd._AUTOCAST.cache[key][2].shape == (3,)
    assert a.dtype == b.dtype == dtype
    (a + b).sum().backward()
    assert lin.bias.grad.dtype == lin.weight.grad.dtype == "float32"
    np.testing.assert_array_equal(lin.bias.grad.numpy(), np.full(3, 10, f32))  # 2 calls x 5 rows
    # the values: the half-precision product plus the half-precision bias
    w16 = lin.weight.numpy().astype(np.float16 if dtype == "float16" else a.numpy().dtype)
    ref = (x.numpy().astype(w16.dtype).astype(f32) @ w16.astype(f32)).astype(w16.dtype).astype(f32)
    ref = ref + lin.bias.numpy().astype(w16.dtype).astype(f32)
    np.testing.assert_allclose(a.numpy().astype(f32), ref, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_linear_without_autocast_unchanged(device):
    x = T((5, 4), device, requires_grad=False)
    lin = nn.Linear(4, 3, device=device)
    out = lin(x)
    assert out.dtype == "float32"
    np.testing.assert_allclose(out.numpy(), x.numpy() @ lin.weight.numpy() + lin.bias.numpy(), rtol=1e-5)
    lin16 = nn.Linear(4, 3, device=device, dtype="float16")  # a float16 model: already float16
    assert lin16(T((5, 4), device, dtype="float16", requires_grad=False)).dtype == "float16"
    with ndl.autocast(enabled=False):
        assert lin(x).dtype == "float32"


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_transformer_fp16_segments_keep_linear_outputs_half(device):
    """Inside "fp16" segments every Linear output (the FFN's, with biases) is
    float16; float32 segments and the output projection stay float32."""
    from models import LanguageModel

    np.random.seed(0)
    m = LanguageModel(16, 30, 32, num_layers=2, seq_model="transformer", device=device,
                      policies=["fp16", "fp32"])
    seen = []
    orig = nn.Linear.forward

    def spy(self, X):
        out = orig(self, X)
        seen.append((self.need_bias, out.dtype))
        return out

    nn.Linear.forward = spy
    try:
        x = ndl.Tensor(np.random.RandomState(1).randint(0, 30, (6, 2)), device=device, requires_grad=False)
        m(x)
    finally:
        nn.Linear.forward = orig
    per_layer = len(seen[:-1]) // 2  # q, k, v, out projections and the two FFN linears
    assert per_layer == 6
    assert all(d == "float16" for _, d in seen[:per_layer])
    assert any(bias for bias, _ in seen[:per_layer])
    assert all(d == "float32" for _, d in seen[per_layer:])


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_autocast_to_helper(device):
    w = T((3,), device)
    assert autograd.autocast_to(w, "float16") is w  # outside autocast: unchanged
    with ndl.autocast():
        a = autograd.autocast_to(w, "float16")
        assert a.dtype == "float16" and a is autograd.autocast_to(w, "float16")  # cached
        assert autograd.autocast_to(w, "float32") is w
        h = T((3,), device, dtype="float16", requires_grad=False)
        assert autograd.autocast_to(h, "float32").dtype == "float32"  # not a parameter: plain cast
