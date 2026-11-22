"""needle.policy.controller: re-planning the segment policies during training."""
import json
import sys

sys.path.append("./python")
sys.path.append(".")
sys.path.append("./tests/solver")
import numpy as np
import pytest

pytest.importorskip("scipy", minversion="1.9")

import needle as ndl
import needle.nn as nn
from apps.models import LanguageModel
from apps.simple_ml import train_ptb
from needle.policy import (PolicyController, SegmentErrors, optimizer_state_bytes, sample_ptb_batches)
from needle.policy import controller as controller_module
from test_solver import make_errors, make_profile

V, SEQ, BS, LAYERS = 60, 12, 4, 3


def make_model(device=None, layers=LAYERS, dropout=0.0):
    np.random.seed(0)
    m = LanguageModel(32, V, 64, num_layers=layers, seq_model="transformer", seq_len=SEQ,
                      device=device or ndl.cpu(), policies="fp32")
    for layer in m.sequence_model.layers.modules:
        layer.dropout.p = dropout
        layer.attn_layer.attn.dropout.p = dropout
    return m


def make_data(steps=6, seed=1):
    return np.random.RandomState(seed).randint(0, V, (SEQ * steps + 1, BS)).astype(np.float32)


def fast(**kw):
    """Controller options that keep the CPU tests quick."""
    base = dict(profile_warmup=0, profile_repeats=1, error_batches=3)
    base.update(kw)
    return base


def params_of(model):
    return [p.numpy().copy() for p in model.parameters()]


# --------------------------------------------------------------------------- helpers


def test_sample_ptb_batches():
    data = make_data(steps=10)
    batches = sample_ptb_batches(data, SEQ, 3)
    assert len(batches) == 3
    for x, y in batches:
        assert x.shape == (SEQ, BS) and y.shape == (SEQ * BS,)
    np.testing.assert_array_equal(batches[0][0].numpy(), data[:SEQ])  # the first ...
    np.testing.assert_array_equal(batches[-1][0].numpy(), data[9 * SEQ:10 * SEQ])  # ... and the last
    assert len(sample_ptb_batches(data, SEQ, 50)) == 10  # only the full-length batches there are
    short = data[: SEQ * 2 + 5]  # 2 full batches and a partial one
    assert len(sample_ptb_batches(short, SEQ, 5)) == 2
    with pytest.raises(ValueError, match="no batch"):
        sample_ptb_batches(data[:SEQ - 2], SEQ, 2)


def test_optimizer_state_bytes():
    m = make_model()
    nbytes = sum(int(np.prod(p.shape)) * 4 for p in m.parameters())
    adam, sgd = ndl.optim.Adam(m.parameters()), ndl.optim.SGD(m.parameters())
    assert optimizer_state_bytes(adam) == 2 * nbytes and optimizer_state_bytes(sgd) == nbytes
    assert optimizer_state_bytes(None) == 0
    assert optimizer_state_bytes(adam, expected=False) == 0  # nothing before the first step
    x, y = sample_ptb_batches(make_data(), SEQ, 1)[0]
    nn.SoftmaxLoss()(m(x)[0], y).backward()
    adam.step()
    assert optimizer_state_bytes(adam, expected=False) == 2 * nbytes


# --------------------------------------------------------------------------- planning during training


def test_replans_every_k_epochs(tmp_path):
    m, data = make_model(), make_data()
    log = tmp_path / "train.json"
    # no budget: an epoch over it would (correctly) add an early re-plan
    ctrl = PolicyController(m, eps=0.02, replan_every=2, log_path=str(log), **fast())
    train_ptb(m, data, seq_len=SEQ, n_epochs=5, optimizer=ndl.optim.Adam, lr=1e-3, controller=ctrl)
    assert [h["epoch"] for h in ctrl.history] == [0, 2, 4]
    assert [h["reason"] for h in ctrl.history] == ["schedule"] * 3
    assert [h["profiled"] for h in ctrl.history] == [True, False, False]  # profiled once, then cached
    for h in ctrl.history:
        assert len(h["segment_errors"]) == LAYERS and h["noise"] > 0 and h["profile_key"] == f"{SEQ}x{BS}"
        assert h["optimizer_state"] == optimizer_state_bytes(ctrl.optimizer)
    assert m.sequence_model.policies == ctrl.history[-1]["policies"] == ctrl.plan.policies
    assert [e["epoch"] for e in ctrl.epochs] == list(range(5)) and all(e["steps"] == 6 for e in ctrl.epochs)
    assert all(e["measured_peak"] > 0 and e["seconds"] > 0 for e in ctrl.epochs)
    saved = json.loads(log.read_text())
    assert saved["config"]["replan_every"] == 2 and saved["config"]["optimizer"] == "Adam"
    assert len(saved["replans"]) == 3 and len(saved["epochs"]) == 5
    assert "epoch 0 (schedule)" in ctrl.summary()


def test_plan_follows_changing_errors(monkeypatch):
    """With a GPU-like profile (float16 faster), the plan uses float16 while the
    estimated errors are small and drops it when they grow."""
    m = make_model()
    scales = iter([0.01, 100.0, 0.01])

    def fake_estimate(transformer, fns, **kw):
        err = make_errors(LAYERS, 0)
        f = next(scales)
        err.grad_sq_batches = [{"fp16": [v * f for v in b["fp16"]]} for b in err.grad_sq_batches]
        return err

    monkeypatch.setattr(controller_module, "estimate_segment_errors", fake_estimate)
    ctrl = PolicyController(m, eps=0.05, replan_every=1, profile=make_profile(LAYERS, 0), **fast())
    batches = sample_ptb_batches(make_data(), SEQ, 3)
    plans = [ctrl.before_epoch(e, batches).policies for e in range(3)]
    assert any(p in ("fp16", "fp16_none") for p in plans[0])
    assert set(plans[1]) <= {"fp32", "none"}
    assert plans[2] == plans[0]
    assert m.sequence_model.policies == plans[2]


def test_memory_feedback_learns_what_the_profile_misses():
    """The optimizer step runs while the step's graph is alive: the measured peak
    of an epoch is above the profile's prediction. The excess is learned and added
    at the next re-plan, whose prediction then covers the measurement. (Recomputing
    every segment keeps the activations small, so the optimizer step sets the peak;
    with only "none" allowed the plan does not depend on the noisy CPU timings.)"""
    m, data = make_model(), make_data()
    ctrl = PolicyController(m, budget_fraction=0.9, policies=["none"], replan_every=2, **fast())
    train_ptb(m, data, seq_len=SEQ, n_epochs=4, optimizer=ndl.optim.Adam, lr=1e-3, controller=ctrl)
    first = ctrl.epochs[0]
    excess = first["measured_peak"] - first["predicted_peak"]
    assert excess > 0 and ctrl.history[0]["unmodeled"] == 0
    assert ctrl.history[1]["unmodeled"] >= excess
    for e in ctrl.epochs[2:]:
        assert e["measured_peak"] <= e["predicted_peak"]


def test_over_budget_replans_early():
    """A feasible plan whose epoch goes over the budget (memory outside the profile)
    makes the next epoch re-plan, without waiting for the schedule."""
    m, data = make_model(), make_data()
    probe = PolicyController(m, policies=["none"], replan_every=100, optimizer=ndl.optim.Adam(m.parameters()),
                             **fast())
    plan = probe.replan(sample_ptb_batches(data, SEQ, 1), epoch=0)
    m2 = make_model()  # its plan is the same; the budget is its predicted peak exactly
    ctrl = PolicyController(m2, budget=plan.peak, policies=["none"], replan_every=100,
                            profile=probe.profile, **fast())
    train_ptb(m2, data, seq_len=SEQ, n_epochs=3, optimizer=ndl.optim.Adam, lr=1e-3, controller=ctrl)
    assert ctrl.epochs[0]["over_budget"]
    assert [h["reason"] for h in ctrl.history][:2] == ["schedule", "over budget"]
    assert ctrl.history[1]["unmodeled"] > 0


def test_infeasible_plan_does_not_replan_every_epoch():
    m, data = make_model(), make_data()
    ctrl = PolicyController(m, budget=1, replan_every=100, **fast())
    train_ptb(m, data, seq_len=SEQ, n_epochs=3, optimizer=ndl.optim.SGD, lr=0.1, controller=ctrl)
    assert len(ctrl.history) == 1 and not ctrl.history[0]["feasible"]
    assert all(e["over_budget"] for e in ctrl.epochs)


# --------------------------------------------------------------------------- planning does not disturb training


@pytest.mark.parametrize("policy", ["none", "fp16"])
def test_training_is_the_same_as_with_the_policies_set_by_hand(policy):
    """Re-planning every epoch (profiling, estimating errors, solving) leaves the
    random state, gradients and parameters alone: the run equals a plain run with
    the chosen policies, bit for bit."""
    data = make_data()
    m = make_model()
    eps = 0.0 if policy == "none" else None  # "none": float32 numerics only
    ctrl = PolicyController(m, eps=eps, policies=[policy], replan_every=1, **fast())
    train_ptb(m, data, seq_len=SEQ, n_epochs=3, optimizer=ndl.optim.Adam, lr=1e-3, controller=ctrl)
    assert all(h["policies"] == [policy] * LAYERS for h in ctrl.history) and len(ctrl.history) == 3

    ref = make_model()
    ref.set_policies(policy)
    scaler = ndl.amp.GradScaler(enabled=policy == "fp16")
    train_ptb(ref, data, seq_len=SEQ, n_epochs=3, optimizer=ndl.optim.Adam, lr=1e-3, scaler=scaler)
    for a, b in zip(params_of(m), params_of(ref)):
        np.testing.assert_array_equal(a, b)
    assert ctrl.scaler.is_enabled() == (policy == "fp16")


def test_replan_keeps_the_random_state():
    """Profiling runs steps in training mode, so dropout draws random numbers; the
    re-plan puts numpy's random state back (train_ptb reseeds every epoch anyway,
    but other training loops may not)."""
    m = make_model(dropout=0.1)
    ctrl = PolicyController(m, eps=0.02, replan_every=1, **fast())
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    ctrl.replan(sample_ptb_batches(make_data(), SEQ, 3), epoch=0)
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_one_scaler_for_the_whole_training():
    m = make_model()
    ctrl = PolicyController(m, replan_every=1, profile=make_profile(LAYERS, 1), **fast())
    scaler = ctrl.scaler
    assert scaler.is_enabled()  # float16 policies allowed
    batches = sample_ptb_batches(make_data(), SEQ, 1)
    for e in range(3):
        ctrl.before_epoch(e, batches)
        assert ctrl.scaler is scaler
    assert not PolicyController(make_model(), eps=0.0, **fast()).scaler.is_enabled()
    assert not PolicyController(make_model(), policies=["fp32", "none"], **fast()).scaler.is_enabled()


# --------------------------------------------------------------------------- modes and options


def test_static_policies_only_log(monkeypatch):
    def no_profiling(*a, **k):
        raise AssertionError("a static controller must not profile")

    monkeypatch.setattr(controller_module, "profile_segments", no_profiling)
    m, data = make_model(), make_data()
    ctrl = PolicyController(m, static_policies=["fp16", "none", "fp32"], **fast())
    assert m.sequence_model.policies == ["fp16", "none", "fp32"] and ctrl.scaler.is_enabled()
    train_ptb(m, data, seq_len=SEQ, n_epochs=2, optimizer=ndl.optim.SGD, lr=0.1, controller=ctrl)
    assert ctrl.history == [] and len(ctrl.epochs) == 2
    assert all(e["policies"] == ["fp16", "none", "fp32"] and e["predicted_peak"] is None for e in ctrl.epochs)
    with pytest.raises(RuntimeError, match="static"):
        ctrl.replan(sample_ptb_batches(data, SEQ, 1))


def test_given_profile_skips_profiling(monkeypatch):
    def no_profiling(*a, **k):
        raise AssertionError("profiled although a profile was given")

    monkeypatch.setattr(controller_module, "profile_segments", no_profiling)
    prof = make_profile(LAYERS, 2)
    ctrl = PolicyController(make_model(), budget_fraction=0.5, profile=prof, **fast())
    plan = ctrl.replan(sample_ptb_batches(make_data(), SEQ, 1), epoch=0)
    assert ctrl.profile is prof and not ctrl.history[0]["profiled"]
    assert plan.budget == pytest.approx(0.5 * prof.predict_peak("fp32"))  # no optimizer given
    prof.key = "256x32"  # a profile of other batches is refused
    with pytest.raises(ValueError, match="256x32"):
        PolicyController(make_model(), profile=prof, **fast()).replan(sample_ptb_batches(make_data(), SEQ, 1))


def test_budget_includes_the_optimizer_state():
    prof = make_profile(LAYERS, 3)
    m = make_model()
    opt = ndl.optim.Adam(m.parameters())
    ctrl = PolicyController(m, budget_fraction=0.5, profile=prof, optimizer=opt, **fast())
    plan = ctrl.replan(sample_ptb_batches(make_data(), SEQ, 1), epoch=0)
    state = optimizer_state_bytes(opt)
    assert ctrl.history[0]["optimizer_state"] == state
    assert plan.peak == prof.predict_peak(plan.policies) + state
    assert plan.budget == pytest.approx(0.5 * (prof.predict_peak("fp32") + state))


def test_argument_checks():
    m = make_model()
    with pytest.raises(ValueError, match="not both"):
        PolicyController(m, budget=1e9, budget_fraction=0.5)
    with pytest.raises(ValueError, match="replan_every"):
        PolicyController(m, replan_every=0)
    with pytest.raises(ValueError, match="error_batches"):
        PolicyController(m, eps=0.01, error_batches=1)
    with pytest.raises(ValueError, match="nn.Transformer"):
        PolicyController(nn.Sequential(nn.Linear(4, 4)))
    with pytest.raises(ValueError, match="exclusive"):
        train_ptb(m, make_data(), seq_len=SEQ, n_epochs=1, amp=True, controller=PolicyController(m))


def test_train_policies_script(tmp_path, capsys):
    from apps.train_policies import main

    common = ["--data", "random", "--device", "cpu", "--layers", "3", "--embedding", "32", "--hidden", "64",
              "--batch", "4", "--seq-len", "12", "--random-steps", "5", "--epochs", "2"]
    log = tmp_path / "run.json"
    ctrl = main(common + ["--budget-fraction", "0.9", "--eps", "0.02", "--replan-every", "1",
                          "--error-batches", "3", "--log", str(log)])
    out = capsys.readouterr().out
    assert "epoch 0 (schedule)" in out and f"log: {log}" in out
    saved = json.loads(log.read_text())
    assert len(saved["replans"]) == 2 and len(saved["epochs"]) == 2
    assert saved["results"]["config"]["num_layers"] == 3 and "train_loss" in saved["results"]
    ctrl = main(common + ["--static", "fp16_none"])
    assert ctrl.history == [] and ctrl.epochs[0]["policies"] == ["fp16_none"] * 3
