"""needle.policy profiler: per-segment time and memory costs of the segment policies."""
import sys
import time

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle.policy import ProfileCache, SegmentProfile, measure_step, profile_segments

_HAS_GPU = ndl.cuda().enabled()
_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
LAYERS = 3


def setup(device, layers=LAYERS, size=(32, 60, 64), shape=(12, 4), policies="fp32"):
    from models import LanguageModel

    emb, vocab, hidden = size
    np.random.seed(0)
    m = LanguageModel(emb, vocab, hidden, num_layers=layers, seq_model="transformer",
                      device=device, policies=policies)
    rng = np.random.RandomState(1)
    x = ndl.Tensor(rng.randint(0, vocab, shape), device=device, requires_grad=False)
    y = ndl.Tensor(rng.randint(0, vocab, shape[0] * shape[1]), device=device, requires_grad=False)
    return m, (lambda: nn.SoftmaxLoss()(m(x)[0], y))


@pytest.fixture(scope="module")
def cpu_profile():
    m, loss_fn = setup(ndl.cpu())
    return m, loss_fn, profile_segments(m.sequence_model, loss_fn, params=m.parameters(),
                                        warmup=0, repeats=1, key="12x4")


# --------------------------------------------------------------------------- plumbing


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_synchronize(device):
    device.synchronize()  # no-op on cpu / numpy, cudaDeviceSynchronize on cuda


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_segment_hook_events(device):
    m, loss_fn = setup(device, policies=["fp32", "fp16", "none"])
    events = []
    m.sequence_model.segment_hook = lambda *e: events.append(e)
    loss_fn()
    assert events == [("start", 0, "fp32"), ("end", 0, "fp32"), ("start", 1, "fp16"),
                      ("end", 1, "fp16"), ("start", 2, "none"), ("end", 2, "none")]
    events.clear()
    m.sequence_model.set_policies(None)  # no segments without policies
    loss_fn()
    assert events == []


def test_profile_structure(cpu_profile):
    _, _, prof = cpu_profile
    assert prof.num_segments == LAYERS and prof.policies == ["fp32", "fp16", "none", "fp16_none"]
    assert prof.device == "cpu" and prof.key == "12x4" and prof.mma is None
    assert prof.half_softmax is False
    assert prof.base_time > 0 and prof.static > 0 and prof.base_peak > 0
    for i in range(LAYERS):
        assert prof.time_delta[i]["fp32"] == 0.0 and prof.peak_delta[i]["fp32"] == 0
        assert set(prof.time_delta[i]) == set(prof.peak_delta[i]) == {"fp32", "fp16", "none", "fp16_none"}
        assert set(prof.recompute[i]) == {"none", "fp16_none"}
        assert all(v >= 0 for v in prof.time_spread[i].values())
    assert set(prof.measured_peak) == {"fp32,fp32,fp32", "fp16,fp16,fp16", "none,none,none",
                                       "fp16_none,fp16_none,fp16_none"}
    assert prof.correction >= 0


def test_memory_costs_are_ordered(cpu_profile):
    _, _, prof = cpu_profile
    for i in range(LAYERS):
        pd, held, rec = prof.peak_delta[i], prof.held[i], prof.recompute[i]
        assert pd["fp16_none"] <= pd["none"] < pd["fp16"] < pd["fp32"] == 0
        assert held["fp16_none"] <= held["none"] < held["fp16"] < held["fp32"]
        # recompute re-materialises the interior, in float16 for "fp16_none"
        assert held["none"] < rec["fp16_none"] < rec["none"]
        assert prof.forward_peak[i]["fp32"] >= prof.held[i]["fp32"]


def test_state_restored(cpu_profile):
    m, loss_fn, _ = cpu_profile
    tr = m.sequence_model
    assert tr.policies == ["fp32"] * LAYERS and tr.segment_hook is None
    # existing gradients, policies and a user's hook survive profiling; the hook
    # is detached while profiling so it cannot disturb the timings
    loss_fn().backward()
    grads = [p.grad for p in m.parameters()]
    tr.set_policies(["none", "fp16", "fp32"])
    calls = []
    user_hook = lambda *e: calls.append(e)
    tr.segment_hook = user_hook
    profile_segments(tr, loss_fn, params=m.parameters(), policies=["fp32", "none"], warmup=0, repeats=1)
    measure_step(tr, loss_fn, "fp16", params=m.parameters(), warmup=0, repeats=1)
    assert calls == []
    assert tr.policies == ["none", "fp16", "fp32"] and tr.segment_hook is user_hook
    assert all(p.grad is g for p, g in zip(m.parameters(), grads))
    tr.segment_hook = None
    tr.set_policies("fp32")


def test_argument_checks(cpu_profile):
    m, loss_fn, prof = cpu_profile
    with pytest.raises(ValueError, match="fp32"):
        profile_segments(m.sequence_model, loss_fn, policies=["fp16", "none"])
    with pytest.raises(ValueError, match="unknown"):
        profile_segments(m.sequence_model, loss_fn, policies=["fp32", "bf16"])
    with pytest.raises(ValueError):
        prof.predict_time(["fp32"] * (LAYERS + 1))
    small = profile_segments(m.sequence_model, loss_fn, params=m.parameters(),
                             policies=["fp32", "none"], warmup=0, repeats=1)
    with pytest.raises(ValueError, match="not profiled"):
        small.predict_peak(["fp16"] * LAYERS)


# --------------------------------------------------------------------------- the models


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_peak_model_matches_measured_steps(device):
    """The additive peak model reproduces the measured peak of complete steps for
    mixed policies it was never calibrated on."""
    m, loss_fn = setup(device)
    prof = profile_segments(m.sequence_model, loss_fn, params=m.parameters(), warmup=0, repeats=1)
    rng = np.random.RandomState(0)
    for _ in range(5):
        pol = list(rng.choice(["fp32", "fp16", "none", "fp16_none"], LAYERS))
        measured = measure_step(m.sequence_model, loss_fn, pol, params=m.parameters(),
                                warmup=0, repeats=1)
        predicted = prof.predict_peak(pol, absolute=False)
        assert predicted >= measured["peak"]  # the correction never under-predicts
        assert predicted <= 1.02 * measured["peak"], (pol, predicted, measured["peak"])
        assert prof.predict_peak(pol) == predicted + prof.static
    # without its correction the model already reproduces the calibration peaks
    assert set(prof.residuals) == set(prof.measured_peak)
    assert max(abs(r) for r in prof.residuals.values()) <= 0.02 * prof.base_peak, prof.residuals


def test_correction_covers_interactions():
    """Memory the single-switch measurements cannot see (here 1 MiB allocated only
    when every segment recomputes) shows up as a calibration residual, and the
    correction lifts the predictions so the peak is not under-predicted."""
    dev = ndl.cpu()
    m, loss_fn = setup(dev)
    tr = m.sequence_model
    extra_bytes = 1 << 20

    def interacting_loss():
        loss = loss_fn()
        if tr.policies == ["none"] * LAYERS:
            # held by the loss tensor, i.e. for the whole step
            loss.extra = ndl.init.zeros(extra_bytes // 4, device=dev, requires_grad=False)
        return loss

    prof = profile_segments(tr, interacting_loss, params=m.parameters(), warmup=0, repeats=1)
    assert prof.residuals["none,none,none"] == pytest.approx(extra_bytes, rel=0.05)
    assert prof.correction == max(prof.residuals.values())
    measured = measure_step(tr, interacting_loss, "none", params=m.parameters(), warmup=0, repeats=1)
    assert prof.predict_peak("none", absolute=False) >= measured["peak"]


def test_time_model(cpu_profile):
    m, loss_fn, prof = cpu_profile
    assert prof.predict_time("fp32") == prof.base_time
    pol = ["none", "fp16", "fp32"]
    expected = prof.base_time + prof.time_delta[0]["none"] + prof.time_delta[1]["fp16"]
    assert prof.predict_time(pol) == pytest.approx(expected)
    # timings are noisy on a shared CPU: only a loose agreement is checked here
    measured = measure_step(m.sequence_model, loss_fn, pol, params=m.parameters(),
                            warmup=1, repeats=3)
    assert 0.5 < prof.predict_time(pol) / measured["time"] < 2.0


def test_time_delta_attributed_to_segment():
    """An extra 60 ms whenever segment 1 recomputes lands in segment 1's "none" delta."""
    m, loss_fn = setup(ndl.cpu())
    tr = m.sequence_model

    def slow_loss():
        if tr.policies[1] == "none":
            time.sleep(0.06)
        return loss_fn()

    prof = profile_segments(tr, slow_loss, params=m.parameters(), policies=["fp32", "none"],
                            warmup=1, repeats=3)
    extra = prof.time_delta[1]["none"] - (prof.time_delta[0]["none"] + prof.time_delta[2]["none"]) / 2
    assert 0.045 < extra < 0.075, prof.summary()
    assert prof.predict_time(["fp32", "none", "fp32"]) - prof.predict_time("fp32") > 0.045


def test_paired_timing_cancels_drift():
    """Every step is 5 ms slower than the one before: timing each configuration in
    its own block would add 20, 40, 60 ms to the deltas of segments 0, 1, 2; the
    paired baseline steps cancel the drift."""
    m, loss_fn = setup(ndl.cpu())
    calls = [0]

    def drifting_loss():
        calls[0] += 1
        time.sleep(0.005 * calls[0])
        return loss_fn()

    prof = profile_segments(m.sequence_model, drifting_loss, params=m.parameters(),
                            policies=["fp32", "none"], warmup=1, repeats=3)
    deltas = [prof.time_delta[i]["none"] for i in range(LAYERS)]
    assert max(deltas) - min(deltas) < 0.015, deltas


def test_recompute_costs_time():
    """'none' replays its segment's forward during backward: over several steps the
    all-'none' step takes longer than the all-'fp32' step."""
    m, loss_fn = setup(ndl.cpu(), size=(64, 60, 256), shape=(32, 8))
    t32 = measure_step(m.sequence_model, loss_fn, "fp32", params=m.parameters(), warmup=1, repeats=5)
    tnone = measure_step(m.sequence_model, loss_fn, "none", params=m.parameters(), warmup=1, repeats=5)
    assert tnone["time"] > t32["time"]
    assert tnone["peak"] < t32["peak"]


def test_custom_calibration_configs():
    m, loss_fn = setup(ndl.cpu())
    configs = [["fp32", "fp16", "none"], ["none", "none", "fp32"]]
    prof = profile_segments(m.sequence_model, loss_fn, params=m.parameters(), warmup=0,
                            repeats=1, calibration=configs)
    assert set(prof.measured_peak) == {"fp32,fp16,none", "none,none,fp32"}


# --------------------------------------------------------------------------- persistence / cache


def test_save_load_round_trip(cpu_profile, tmp_path):
    _, _, prof = cpu_profile
    path = tmp_path / "profile.json"
    prof.save(str(path))
    loaded = SegmentProfile.load(str(path))
    assert loaded == prof
    for pol in (["fp32", "fp16", "none"], "none", "fp16_none"):
        assert loaded.predict_peak(pol) == prof.predict_peak(pol)
        assert loaded.predict_time(pol) == prof.predict_time(pol)
    assert "segment 0" in prof.summary() and "recompute fp16_none" in prof.summary()
    # profiles saved before "fp16_none" stored one recompute number per segment
    old = prof.to_dict()
    old["policies"] = ["fp32", "fp16", "none"]
    old["recompute"] = [r["none"] for r in prof.recompute]
    del old["half_softmax"]
    legacy = SegmentProfile.from_dict(old)
    assert legacy.recompute == [{"none": r["none"]} for r in prof.recompute]
    assert legacy.half_softmax is None and "recompute none" in legacy.summary()


def test_half_softmax_recorded_and_cheaper():
    m, loss_fn = setup(ndl.cpu(), shape=(48, 4))
    tr = m.sequence_model
    profs = {}
    for flag in (False, True):
        tr.half_softmax = flag
        profs[flag] = profile_segments(tr, loss_fn, params=m.parameters(), policies=["fp32", "fp16"],
                                       warmup=0, repeats=1)
        assert profs[flag].half_softmax is flag
    tr.half_softmax = False
    for i in range(LAYERS):
        assert profs[True].peak_delta[i]["fp16"] < profs[False].peak_delta[i]["fp16"]
    assert profs[True].base_peak == profs[False].base_peak  # float32 segments unaffected


def test_profile_cache(cpu_profile):
    _, _, prof = cpu_profile
    cache = ProfileCache()
    calls = []

    def make():
        calls.append(1)
        return prof

    assert cache.get_or_profile((12, 4), make) is prof
    assert cache.get_or_profile((12, 4), make) is prof
    assert calls == [1] and (12, 4) in cache and len(cache) == 1
    assert cache.get((32, 8)) is None
    cache.put((32, 8), prof)
    assert len(cache) == 2


# --------------------------------------------------------------------------- GPU


@pytest.mark.skipif(not _HAS_GPU, reason="No GPU")
def test_gpu_profile_uses_tensor_cores_and_restores_switch():
    dev = ndl.cuda()
    m, loss_fn = setup(dev, size=(64, 100, 256), shape=(64, 8))
    before = dev.mma_enabled()
    dev.set_mma_enabled(False)
    try:
        prof = profile_segments(m.sequence_model, loss_fn, params=m.parameters(), warmup=1, repeats=3)
        assert prof.device == "cuda"
        assert prof.mma is dev.mma_available("float16")  # tensor cores used where supported
        assert dev.mma_enabled() is False  # restored
        off = profile_segments(m.sequence_model, loss_fn, params=m.parameters(), warmup=1,
                               repeats=3, mma=False)
        assert off.mma is False
        print("\n" + prof.summary())
        print("fp16 time delta per segment, tensor cores on / off (ms):",
              [round(prof.time_delta[i]["fp16"] * 1e3, 3) for i in range(LAYERS)], "/",
              [round(off.time_delta[i]["fp16"] * 1e3, 3) for i in range(LAYERS)])
    finally:
        dev.set_mma_enabled(before)
