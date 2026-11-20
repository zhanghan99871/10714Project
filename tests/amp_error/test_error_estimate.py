"""needle.policy.error: first-order estimate of the loss error of half-precision segments."""
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle.nn.nn_transformer import SEGMENT_POLICIES
from needle.policy import SegmentErrors, estimate_segment_errors, measure_precision_error
from needle.policy.error import _Preserved, _reference_pass, _rerun

_HAS_GPU = ndl.cuda().enabled()
_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
LAYERS = 3


def setup(device, size=(32, 60, 64), shape=(12, 4), batches=2, dropout=0.0, half_softmax=False):
    from models import LanguageModel

    emb, vocab, hidden = size
    np.random.seed(0)
    m = LanguageModel(emb, vocab, hidden, num_layers=LAYERS, seq_model="transformer", device=device,
                      policies="fp32", half_softmax=half_softmax)
    for layer in m.sequence_model.layers.modules:
        layer.dropout.p = dropout
        layer.attn_layer.attn.dropout.p = dropout
    fns = []
    for b in range(batches):
        rng = np.random.RandomState(10 + b)
        x = ndl.Tensor(rng.randint(0, vocab, shape), device=device, requires_grad=False)
        y = ndl.Tensor(rng.randint(0, vocab, shape[0] * shape[1]), device=device, requires_grad=False)
        fns.append(lambda x=x, y=y: nn.SoftmaxLoss()(m(x)[0], y))
    return m, fns


def single(i, policy="fp16"):
    pol = ["fp32"] * LAYERS
    pol[i] = policy
    return pol


@pytest.fixture(scope="module")
def cpu_estimate():
    m, fns = setup(ndl.cpu())
    return m, fns, estimate_segment_errors(m.sequence_model, fns, params=m.parameters())


# --------------------------------------------------------------------------- structure


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_error_table(device):
    m, fns = setup(device)
    est = estimate_segment_errors(m.sequence_model, fns, params=m.parameters())
    assert est.policies == list(SEGMENT_POLICIES) and est.num_segments == LAYERS
    assert est.reference == "none" and est.half_softmax is False and len(est.loss) == 2
    for i in range(LAYERS):
        assert est.error[i]["fp32"] == est.error[i]["none"] == 0.0
        # "fp16_none" recomputes the same float16 segment: the same error
        assert est.error[i]["fp16"] == est.error[i]["fp16_none"] > 0
        assert 0 < est.perturbation[i]["fp16"] < 1e-2  # |d| / |o|: float16 rounding
    assert est.predict("fp32") == est.predict("none") == est.predict_signed("none") == 0.0
    pol = ["fp16", "none", "fp16_none"]
    assert est.predict(pol) == pytest.approx(est.error[0]["fp16"] + est.error[2]["fp16"])
    assert est.predict(pol, relative=True) == pytest.approx(est.predict(pol) / est.mean_loss)
    # |sum_i s_i| <= sum_i |s_i| per batch, so the signed estimate never exceeds the bound
    assert est.predict_signed(pol) <= est.predict(pol) + 1e-15
    assert est.predict_signed(single(1)) == pytest.approx(est.predict(single(1)))


@pytest.mark.parametrize("reference", ["none", "fp32"])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_rerun_reproduces_the_reference_exactly(device, reference):
    """The perturbation is due to precision alone: rerunning a segment in float32
    from its recorded input reproduces the reference output bit for bit."""
    m, fns = setup(device)
    tr = m.sequence_model
    with _Preserved(tr, m.parameters(), True) as state:
        _, xs, outs, grads = _reference_pass(tr, fns[0], reference, state)
        for i in range(LAYERS):
            for policy in ("fp32", "none"):
                np.testing.assert_array_equal(_rerun(tr, i, xs[i], policy), outs[i])
            assert not np.array_equal(_rerun(tr, i, xs[i], "fp16"), outs[i])
            assert np.any(grads[i] != 0)


def test_reference_fp32_and_none_agree():
    m, fns = setup(ndl.cpu())
    a = estimate_segment_errors(m.sequence_model, fns, params=m.parameters(), reference="none")
    b = estimate_segment_errors(m.sequence_model, fns, params=m.parameters(), reference="fp32")
    assert a.signed == b.signed and a.loss == b.loss


# --------------------------------------------------------------------------- accuracy


@pytest.mark.parametrize("half_softmax", [False, True])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_first_order_matches_single_segment_change(device, half_softmax):
    """s_i = <grad_{o_i} L, o_i^fp16 - o_i^fp32> predicts the actual loss change of
    switching only segment i to float16, sign included, on every batch."""
    m, fns = setup(device, half_softmax=half_softmax)
    est = estimate_segment_errors(m.sequence_model, fns, params=m.parameters())
    for i in range(LAYERS):
        actual = np.array(measure_precision_error(m.sequence_model, fns, single(i), params=m.parameters(),
                                                  gradients=False)["signed"])
        first_order = np.array([b["fp16"][i] for b in est.signed])
        assert np.all(np.abs(first_order - actual) <= 0.1 * np.abs(actual).max()), (i, first_order, actual)


def test_bound_ranks_configurations_like_measured_errors(cpu_estimate):
    """Over random configurations the linear bound orders the configurations like
    the measured loss error and the measured gradient error."""
    m, fns, est = cpu_estimate
    rng = np.random.RandomState(3)
    configs = [list(rng.choice(est.policies, LAYERS)) for _ in range(10)] + [["fp16"] * LAYERS]
    bound, loss_err, grad_err = [], [], []
    for c in configs:
        r = measure_precision_error(m.sequence_model, fns, c, params=m.parameters())
        bound.append(est.predict(c))
        loss_err.append(r["loss_error"])
        grad_err.append(r["grad_error"])
        if est.predict(c) == 0:
            assert r["loss_error"] == 0 and r["grad_error"] == 0  # float32 numerics only

    def rank_corr(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]

    assert rank_corr(bound, loss_err) > 0.8, (bound, loss_err)
    assert rank_corr(bound, grad_err) > 0.8, (bound, grad_err)
    # a first-order bound: second-order terms may exceed it a little, never by much
    assert all(a <= 1.5 * b + 1e-7 for a, b in zip(loss_err, bound))


def test_multiple_batches_are_averaged():
    m, fns = setup(ndl.cpu())
    both = estimate_segment_errors(m.sequence_model, fns, params=m.parameters())
    each = [estimate_segment_errors(m.sequence_model, fn, params=m.parameters()) for fn in fns]
    assert both.loss == [e.loss[0] for e in each]
    assert both.signed == [e.signed[0] for e in each]
    for i in range(LAYERS):
        assert both.error[i]["fp16"] == pytest.approx(np.mean([e.error[i]["fp16"] for e in each]))


def test_coefficients_average_absolute_changes():
    """The coefficient is the mean of |s_i| over batches, not |mean s_i|: a segment
    whose error changes sign from batch to batch still costs error."""
    m, fns = setup(ndl.cpu(), batches=4)
    est = estimate_segment_errors(m.sequence_model, fns, params=m.parameters())
    s = np.array([b["fp16"] for b in est.signed])  # [batch, segment]
    assert np.any((s > 0).any(axis=0) & (s < 0).any(axis=0))  # signs do change
    for i in range(LAYERS):
        assert est.error[i]["fp16"] == pytest.approx(np.mean(np.abs(s[:, i])))


def test_predict_formulas():
    """predict: sum of the coefficients; predict_signed: per batch |sum_i s_i| over the
    float16 segments, averaged over batches."""
    signed = [{"fp16": [1e-3, -2e-3, 4e-4]}, {"fp16": [1e-3, 1e-3, -4e-4]}]
    error = [{p: (np.mean([abs(b["fp16"][i]) for b in signed]) if "fp16" in p else 0.0)
              for p in SEGMENT_POLICIES} for i in range(3)]
    est = SegmentErrors(policies=list(SEGMENT_POLICIES), num_segments=3, device="cpu", reference="none",
                        half_softmax=False, loss=[2.0, -4.0], signed=signed, error=error)
    pol = ["fp16", "fp16_none", "none"]
    assert est.predict(pol) == pytest.approx(1e-3 + 1.5e-3)
    assert est.predict(pol, relative=True) == pytest.approx(2.5e-3 / 3.0)  # mean |loss| = 3
    assert est.predict_signed(pol) == pytest.approx((1e-3 + 2e-3) / 2)  # |-1e-3|, |2e-3|
    assert est.predict_signed(["fp32", "fp32", "fp16"]) == pytest.approx(4e-4)


# --------------------------------------------------------------------------- measured errors


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_measure_precision_error(device):
    m, fns = setup(device)
    tr = m.sequence_model
    for exact in ("fp32", "none", ["none", "fp32", "none"]):
        r = measure_precision_error(tr, fns, exact, params=m.parameters())
        assert r["loss_error"] == 0.0 and r["grad_error"] == 0.0
    a = measure_precision_error(tr, fns, "fp16", params=m.parameters())
    b = measure_precision_error(tr, fns, "fp16_none", params=m.parameters())
    assert a == b  # bit-identical losses and gradients
    assert 0 < a["loss_error"] < 1e-2 and 0 < a["grad_error"] < 0.1
    c = measure_precision_error(tr, fns, "fp16", params=m.parameters(), gradients=False)
    assert c["signed"] == a["signed"] and c["grad_error"] is None


# --------------------------------------------------------------------------- state and options


def test_state_restored_and_dropout_off():
    m, fns = setup(ndl.cpu(), dropout=0.3)
    tr = m.sequence_model
    m.train()
    fns[0]().backward()
    grads = [p.grad for p in m.parameters()]
    tr.set_policies(["none", "fp16", "fp32"])
    calls = []
    user_hook = lambda *e: calls.append(e)
    tr.segment_hook = user_hook
    a = estimate_segment_errors(tr, fns, params=m.parameters())
    b = estimate_segment_errors(tr, fns, params=m.parameters())
    measure_precision_error(tr, fns, "fp16", params=m.parameters())
    # dropout is off while estimating: the estimate is deterministic despite p = 0.3
    assert a.signed == b.signed
    assert calls == [] and tr.segment_hook is user_hook
    assert tr.policies == ["none", "fp16", "fp32"]
    assert all(p.grad is g for p, g in zip(m.parameters(), grads))
    assert tr.training and all(mod.training for mod in tr._children())


def test_half_softmax_recorded(cpu_estimate):
    m, fns, est = cpu_estimate
    tr = m.sequence_model
    tr.half_softmax = True
    try:
        hs = estimate_segment_errors(tr, fns, params=m.parameters())
    finally:
        tr.half_softmax = False
    assert est.half_softmax is False and hs.half_softmax is True
    assert hs.signed != est.signed  # a different float16 function


def test_argument_checks(cpu_estimate):
    m, fns, est = cpu_estimate
    with pytest.raises(ValueError, match="float32 policy"):
        estimate_segment_errors(m.sequence_model, fns, reference="fp16")
    with pytest.raises(ValueError, match="no loss_fn"):
        estimate_segment_errors(m.sequence_model, [])
    with pytest.raises(ValueError, match="expected"):
        est.predict(["fp16"] * (LAYERS + 1))
    with pytest.raises(ValueError, match="unknown policy"):
        est.predict(["bf16"] * LAYERS)


def test_save_load_round_trip(cpu_estimate, tmp_path):
    _, _, est = cpu_estimate
    path = tmp_path / "errors.json"
    est.save(str(path))
    loaded = SegmentErrors.load(str(path))
    assert loaded == est
    for pol in ("fp16", ["fp16", "none", "fp16_none"]):
        assert loaded.predict(pol) == est.predict(pol)
        assert loaded.predict_signed(pol) == est.predict_signed(pol)
    text = est.summary()
    assert "segment 0" in text and "fp16_none" in text
