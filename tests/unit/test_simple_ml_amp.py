"""The `amp` option of the training loops in apps/simple_ml.py."""
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle.data.datasets.ptb_dataset import batchify

from models import LanguageModel, ResNet9
from simple_ml import (
    epoch_general_cifar10,
    epoch_general_ptb,
    evaluate_cifar10,
    evaluate_ptb,
    train_cifar10,
    train_ptb,
)

DEV = ndl.cpu()


def ptb_data():
    np.random.seed(0)
    return batchify(list(np.random.randint(0, 40, 1200)), 8, DEV, "float32")


def cifar_loader(n=16, batch=8):
    np.random.seed(0)
    ds = ndl.data.NDArrayDataset(
        np.random.rand(n, 3, 32, 32).astype(np.float32), np.random.randint(0, 10, n)
    )
    return ndl.data.DataLoader(ds, batch)


def lm(kind):
    np.random.seed(0)
    return LanguageModel(16, 40, 32, num_layers=1, seq_model=kind, device=DEV)


def resnet():
    np.random.seed(0)
    return ResNet9(device=DEV)


class SpyScaler(ndl.amp.GradScaler):
    """Records how the training loop drives the scaler."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def scale(self, loss):
        self.calls.append("scale")
        return super().scale(loss)

    def unscale_(self, opt):
        self.calls.append("unscale_")
        return super().unscale_(opt)

    def step(self, opt):
        self.calls.append("step")
        return super().step(opt)

    def update(self, new_scale=None):
        self.calls.append("update")
        return super().update(new_scale)


# --------------------------------------------------------------------------- default is unchanged


def test_amp_false_is_the_float32_path():
    """amp=False (the default) enters no autocast region and steps every time."""
    data = ptb_data()
    a = train_ptb(lm("lstm"), data, seq_len=10, lr=0.5, device=DEV)
    b = train_ptb(lm("lstm"), data, seq_len=10, lr=0.5, device=DEV, amp=False)
    assert a == b
    sc = SpyScaler(enabled=False)
    model = lm("rnn")
    epoch_general_ptb(data, model, seq_len=10, opt=ndl.optim.SGD(model.parameters(), lr=0.5),
                      device=DEV, scaler=sc)
    assert sc.skipped_steps == 0 and sc.get_scale() == 1.0


# --------------------------------------------------------------------------- PTB


@pytest.mark.parametrize("kind", ["rnn", "lstm", "transformer"])
def test_train_and_evaluate_ptb_with_amp(kind):
    data = ptb_data()
    lr = 0.05 if kind == "transformer" else 0.5
    res = {}
    for amp in (False, True):
        model = lm(kind)
        tr = train_ptb(model, data, seq_len=10, n_epochs=2, lr=lr, clip=0.25, device=DEV, amp=amp)
        ev = evaluate_ptb(model, data, seq_len=10, device=DEV, amp=amp)
        assert all(p.dtype == "float32" for p in model.parameters())
        assert np.isfinite(tr[1]) and np.isfinite(ev[1])
        res[amp] = (tr, ev)
    (tr32, ev32), (tr16, ev16) = res[False], res[True]
    assert abs(tr16[1] - tr32[1]) < 5e-3 * tr32[1] and abs(ev16[1] - ev32[1]) < 5e-3 * ev32[1]


def test_ptb_amp_scaler_protocol_and_clipping():
    data = ptb_data()
    model = lm("lstm")
    opt = ndl.optim.SGD(model.parameters(), lr=0.5)
    sc = SpyScaler()
    epoch_general_ptb(data, model, seq_len=50, opt=opt, clip=0.25, device=DEV, amp=True, scaler=sc)
    n_batches = len(range(0, data.shape[0] - 1, 50))
    assert sc.calls == ["scale", "unscale_", "step", "update"] * n_batches
    sc = SpyScaler()
    epoch_general_ptb(data, lm("lstm"), seq_len=50, opt=ndl.optim.SGD(model.parameters(), lr=0.5),
                      device=DEV, amp=True, scaler=sc)  # no clip: step() unscales itself
    assert sc.calls == ["scale", "step", "unscale_", "update"] * n_batches


def test_ptb_evaluation_never_scales():
    sc = SpyScaler()
    evaluate_ptb(lm("rnn"), ptb_data(), seq_len=10, device=DEV, amp=True)
    epoch_general_ptb(ptb_data(), lm("rnn"), seq_len=10, device=DEV, amp=True, scaler=sc)
    assert sc.calls == []


def test_overflowing_scale_skips_steps_and_recovers():
    """A scale far too large: early steps are skipped (parameters untouched) while
    the scale backs off; afterwards training proceeds."""
    data = ptb_data()
    model = lm("lstm")
    before = [p.numpy().copy() for p in model.parameters()]
    # this LSTM's float16 gradients fit up to a scale of ~2**21: from 2**30 the
    # first ~9 of the 15 steps overflow, the rest run
    sc = ndl.amp.GradScaler(init_scale=2.0 ** 30)
    opt = ndl.optim.SGD(model.parameters(), lr=0.5)
    epoch_general_ptb(data, model, seq_len=10, opt=opt, device=DEV, amp=True, scaler=sc)
    n_batches = len(range(0, data.shape[0] - 1, 10))
    assert 0 < sc.skipped_steps < n_batches and sc.get_scale() < 2.0 ** 30
    assert any(not np.array_equal(p.numpy(), b) for p, b in zip(model.parameters(), before))


# --------------------------------------------------------------------------- CIFAR


def test_train_and_evaluate_cifar10_with_amp():
    res = {}
    for amp in (False, True):
        model = resnet()
        # a scale that does not overflow, so both runs take the same 4 optimizer steps
        sc = ndl.amp.GradScaler(init_scale=2.0 ** 10, enabled=amp)
        tr = train_cifar10(model, cifar_loader(), n_epochs=2, amp=amp, scaler=sc)
        ev = evaluate_cifar10(model, cifar_loader(), amp=amp)
        assert all(p.dtype == "float32" for p in model.parameters())
        assert np.isfinite(tr[1]) and np.isfinite(ev[1]) and sc.skipped_steps == 0
        res[amp] = (tr, ev)
    assert abs(res[True][0][1] - res[False][0][1]) < 5e-2 * res[False][0][1]


def test_cifar10_default_scaler_may_skip_warmup_steps():
    """With the default init_scale 2**16 the first ResNet9 step overflows in float16
    and is skipped, the scale halves, and training continues."""
    model = resnet()
    sc = ndl.amp.GradScaler()
    opt = ndl.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        acc, loss = epoch_general_cifar10(cifar_loader(), model, nn.SoftmaxLoss(), opt,
                                          amp=True, scaler=sc)
    assert np.isfinite(loss)
    assert sc.skipped_steps >= 1 and sc.get_scale() < 2.0 ** 16
    assert sc.skipped_steps < 4  # not every step


def test_cifar10_eval_mode_under_amp():
    model = resnet()
    acc, loss = evaluate_cifar10(model, cifar_loader(), amp=True)
    assert not model.training and np.isfinite(loss) and 0.0 <= acc <= 1.0


@pytest.mark.parametrize("amp", [False, True])
def test_forward_runs_under_autocast_only_with_amp(amp):
    """amp=True must really run the model (and only the forward) under autocast."""
    seen = []
    model = lm("rnn")
    forward = model.forward

    def spy(*args, **kwargs):
        seen.append(ndl.is_autocast_enabled())
        return forward(*args, **kwargs)

    model.forward = spy
    epoch_general_ptb(ptb_data(), model, seq_len=50, opt=ndl.optim.SGD(model.parameters(), lr=0.1),
                      device=DEV, amp=amp)
    assert seen and all(s == amp for s in seen)
    assert not ndl.is_autocast_enabled()  # nothing leaks out of the loop
