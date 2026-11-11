"""ndl.amp.GradScaler: dynamic loss scaling for float16 mixed precision."""
import sys

sys.path.append("./python")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import ops

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f16, f32 = np.float16, np.float32


def T(x, device, dtype=None, requires_grad=True):
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def _tiny_grad_problem(device):
    """dloss/dW = 3e-8 per entry: below float16's range unless the loss is scaled."""
    W = T(np.ones((4, 2), f32), device)
    x = T(np.full((3, 4), 1e-4, f16), device, requires_grad=False)
    with ndl.autocast():
        loss = ops.summation(x @ W) * 1e-4
    return W, loss


def _overflow_problem(device, scale):
    """float16 gradients overflow once the loss is multiplied by `scale`."""
    W = T(np.ones((2, 2), f32), device)
    x = T(np.full((1, 2), 100.0, f16), device, requires_grad=False)
    with ndl.autocast():
        loss = ops.summation(x @ W)
    return W, loss


# --------------------------------------------------------------------------- basics


def test_constructor_validation():
    for kwargs in (
        {"init_scale": 0}, {"growth_factor": 1.0}, {"backoff_factor": 1.0},
        {"backoff_factor": 0.0}, {"growth_interval": 0},
    ):
        with pytest.raises(ValueError):
            ndl.amp.GradScaler(**kwargs)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_disabled_scaler_is_a_pass_through(device):
    W = T(np.ones((2, 2), f32), device)
    opt = ndl.optim.SGD([W], lr=0.5)
    sc = ndl.amp.GradScaler(enabled=False)
    loss = ops.summation(W * 2.0)
    assert sc.scale(loss) is loss and sc.get_scale() == 1.0 and not sc.is_enabled()
    loss.backward()
    sc.unscale_(opt)  # no-op
    assert sc.step(opt) is True
    sc.update()
    np.testing.assert_array_equal(W.numpy(), np.zeros((2, 2), f32))  # 1 - 0.5 * 2


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_scale_multiplies_in_float32(device):
    sc = ndl.amp.GradScaler(init_scale=1024.0)
    loss16 = T(np.array([2.0], f16), device) * 1.0
    scaled = sc.scale(loss16)
    assert scaled.dtype == "float32" and scaled.numpy()[0] == 2048.0
    assert loss16.cached_data is not None  # the caller's loss is not released
    big = sc.scale(T(np.array([100.0], f16), device) * 1.0)  # 102400 > float16 max
    assert big.numpy()[0] == 102400.0


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_unscaled_gradients_equal_unscaled_backward(device):
    """In float32, scaling by a power of two and unscaling is exact."""
    grads = []
    for use in (False, True):
        np.random.seed(0)
        model = nn.Sequential(nn.Linear(5, 7, device=device), nn.ReLU(), nn.Linear(7, 3, device=device))
        X = T(np.random.randn(4, 5).astype(f32), device, requires_grad=False)
        y = T(np.array([0, 1, 2, 1]), device, requires_grad=False)
        opt = ndl.optim.SGD(model.parameters(), lr=0.1)
        sc = ndl.amp.GradScaler(init_scale=2.0 ** 12, enabled=use)
        sc.scale(nn.SoftmaxLoss()(model(X), y)).backward()
        sc.unscale_(opt)
        grads.append([p.grad.numpy() for p in model.parameters()])
    for a, b in zip(*grads):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_scaling_rescues_float16_underflow(device):
    exact = 3e-8
    W, loss = _tiny_grad_problem(device)
    loss.backward()
    unscaled = W.grad.numpy()
    W, loss = _tiny_grad_problem(device)
    sc, opt = ndl.amp.GradScaler(), ndl.optim.SGD([W], lr=1.0)
    sc.scale(loss).backward()
    sc.unscale_(opt)
    assert W.grad.dtype == "float32"
    assert abs(unscaled[0, 0] - exact) / exact > 0.5  # float16 subnormal: badly off
    np.testing.assert_allclose(W.grad.numpy(), np.full((4, 2), exact), rtol=2e-3)


# --------------------------------------------------------------------------- overflow handling


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_overflow_skips_step_and_backs_off(device):
    W, loss = _overflow_problem(device, 2.0 ** 20)
    opt = ndl.optim.SGD([W], lr=0.1)
    sc = ndl.amp.GradScaler(init_scale=2.0 ** 20)  # 100 * 2**20 overflows float16
    sc.scale(loss).backward()
    assert sc.step(opt) is False
    np.testing.assert_array_equal(W.numpy(), np.ones((2, 2), f32))  # untouched
    sc.update()
    assert sc.get_scale() == 2.0 ** 19 and sc.skipped_steps == 1


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_scale_converges_to_a_working_value(device):
    W = T(np.ones((2, 2), f32), device)
    opt = ndl.optim.SGD([W], lr=1e-3)
    x = T(np.full((1, 2), 100.0, f16), device, requires_grad=False)
    sc = ndl.amp.GradScaler(init_scale=2.0 ** 24, growth_interval=1000)
    stepped = []
    for _ in range(20):
        opt.reset_grad()
        with ndl.autocast():
            loss = ops.summation(x @ W)
        sc.scale(loss).backward()
        stepped.append(sc.step(opt))
        sc.update()
    # overflows until 100 * scale < 65504, i.e. scale <= 2**9; then every step runs
    assert stepped[:15] == [False] * 15 and all(stepped[15:])
    assert sc.get_scale() == 2.0 ** 9 and sc.skipped_steps == 15


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_nan_gradients_are_detected(device):
    W = T(np.array([1.0, -1.0], f32), device)
    opt = ndl.optim.SGD([W], lr=0.1)
    sc = ndl.amp.GradScaler(init_scale=1.0)
    loss = ops.summation(W * float("nan"))  # d loss / dW = NaN
    sc.scale(loss).backward()
    assert sc.step(opt) is False
    sc.update()
    assert sc.get_scale() == 0.5


def test_growth_after_interval_and_reset_on_overflow():
    dev = ndl.cpu()
    sc = ndl.amp.GradScaler(init_scale=8.0, growth_interval=3)
    W = T(np.ones(2, f32), dev)
    opt = ndl.optim.SGD([W], lr=0.0)

    def iteration(value):
        opt.reset_grad()
        sc.scale(ops.summation(W * value)).backward()
        sc.step(opt)
        sc.update()

    for _ in range(2):
        iteration(1.0)
    assert sc.get_scale() == 8.0
    iteration(1.0)
    assert sc.get_scale() == 16.0  # 3 clean steps -> grow
    iteration(1.0)
    iteration(1.0)
    iteration(float("inf"))  # overflow resets the clean-step counter
    assert sc.get_scale() == 8.0
    iteration(1.0)
    iteration(1.0)
    assert sc.get_scale() == 8.0
    iteration(1.0)
    assert sc.get_scale() == 16.0


# --------------------------------------------------------------------------- unscale_ / step protocol


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_unscale_once_and_clip_true_gradients(device):
    W = T(np.array([3.0, 4.0], f32), device)
    opt = ndl.optim.SGD([W], lr=1.0)
    sc = ndl.amp.GradScaler(init_scale=2.0 ** 10)
    sc.scale(ops.summation(W * W) * 0.5).backward()  # grad = W = (3, 4), norm 5
    sc.unscale_(opt)
    np.testing.assert_array_equal(W.grad.numpy(), np.array([3.0, 4.0], f32))
    with pytest.raises(RuntimeError, match="already been called"):
        sc.unscale_(opt)
    opt.clip_grad_norm(1.0)  # sees the true norm 5, not 5 * 1024
    np.testing.assert_allclose(W.grad.numpy(), [0.6, 0.8], rtol=1e-5)
    assert sc.step(opt) is True  # does not divide by the scale a second time
    np.testing.assert_allclose(W.numpy(), [2.4, 3.2], rtol=1e-5)
    sc.update()
    sc.scale(ops.summation(W)).backward()
    sc.unscale_(opt)  # allowed again after update()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_unscale_detaches_gradient_graph(device):
    W = T(np.ones((2, 2), f32), device)
    X = T(np.ones((3, 2), f32), device)  # requires grad: W.grad is a graph node
    opt = ndl.optim.SGD([W], lr=0.1)
    sc = ndl.amp.GradScaler(init_scale=4.0)
    sc.scale(ops.summation(X @ W)).backward(create_graph=True)  # .grad is a graph node
    assert not W.grad.is_leaf()
    sc.unscale_(opt)
    assert W.grad.is_leaf() and not W.grad.requires_grad


def test_multiple_optimizers_back_off_if_any_overflows():
    dev = ndl.cpu()
    A = T(np.ones(2, f32), dev)
    B = T(np.ones(2, f32), dev)
    opt_a, opt_b = ndl.optim.SGD([A], lr=0.1), ndl.optim.SGD([B], lr=0.1)
    sc = ndl.amp.GradScaler(init_scale=4.0)
    sc.scale(ops.summation(A) + ops.summation(B * float("inf"))).backward()
    assert sc.step(opt_a) is True and sc.step(opt_b) is False
    sc.update()
    assert sc.get_scale() == 2.0 and sc.skipped_steps == 1
    np.testing.assert_array_equal(B.numpy(), np.ones(2, f32))


def test_adam_state_untouched_by_skipped_step():
    dev = ndl.cpu()
    W = T(np.ones(2, f32), dev)
    opt = ndl.optim.Adam([W], lr=0.1)
    sc = ndl.amp.GradScaler(init_scale=1.0)
    sc.scale(ops.summation(W * float("inf"))).backward()
    sc.step(opt)
    sc.update()
    assert opt.t == 0 and not opt.m and not opt.v
    np.testing.assert_array_equal(W.numpy(), np.ones(2, f32))


def test_state_dict_round_trip():
    sc = ndl.amp.GradScaler(init_scale=2.0 ** 8, growth_interval=7, backoff_factor=0.25)
    sc.skipped_steps = 3
    sc._growth_tracker = 5
    sc2 = ndl.amp.GradScaler()
    sc2.load_state_dict(sc.state_dict())
    assert sc2.state_dict() == sc.state_dict()
    sc2.update(new_scale=32.0)
    assert sc2.get_scale() == 32.0
    with pytest.raises(ValueError):
        sc2.update(new_scale=0.0)


# --------------------------------------------------------------------------- training


def test_amp_training_with_scaler_tracks_float32():
    dev = ndl.cpu()
    curves = {}
    for amp in (False, True):
        np.random.seed(5)
        X = T(np.random.randn(32, 10).astype(f32), dev, requires_grad=False)
        y = T(np.random.randint(0, 4, 32), dev, requires_grad=False)
        model = nn.Sequential(nn.Linear(10, 32, device=dev), nn.ReLU(), nn.Linear(32, 4, device=dev))
        opt = ndl.optim.Adam(model.parameters(), lr=0.01)
        sc = ndl.amp.GradScaler(init_scale=2.0 ** 10, enabled=amp)
        losses = []
        for _ in range(25):
            opt.reset_grad()
            with ndl.autocast(enabled=amp):
                loss = nn.SoftmaxLoss()(model(X), y)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
            losses.append(loss.numpy().item())
        curves[amp] = np.array(losses)
        assert all(p.dtype == "float32" for p in model.parameters())
    assert curves[False][-1] < 0.6 * curves[False][0]
    np.testing.assert_allclose(curves[True], curves[False], rtol=2e-2, atol=5e-3)
