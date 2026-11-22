"""needle.policy.error: precision errors of the segment policies (first-order loss
change and measured gradient change), combined as sums of squares and measured
against the batch noise."""
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


def setup(device, size=(32, 60, 64), shape=(12, 4), batches=3, dropout=0.0, half_softmax=False):
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


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


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
    assert est.reference == "none" and est.half_softmax is False and len(est.loss) == 3
    assert est.loss_noise > 0 and est.grad_noise > 0 and est.grad_norm_sq > 0
    for metric in ("loss", "grad"):
        sq, coef = est.mean_sq(metric), est.coefficients(metric)
        for i in range(LAYERS):
            assert sq[i]["fp32"] == sq[i]["none"] == 0.0
            # "fp16_none" recomputes the same float16 segment: the same errors
            assert sq[i]["fp16"] == sq[i]["fp16_none"] > 0
            assert coef[i]["fp16"] == pytest.approx(sq[i]["fp16"] / est.noise(metric))
        pol = ["fp16", "none", "fp16_none"]
        assert est.predict(pol, metric, relative=False) == pytest.approx(np.sqrt(sq[0]["fp16"] + sq[2]["fp16"]))
        assert est.predict(pol, metric) == pytest.approx(np.sqrt(coef[0]["fp16"] + coef[2]["fp16"]))
        assert est.predict("fp32", metric) == est.predict("none", metric) == 0.0
    for i in range(LAYERS):
        assert 0 < est.perturbation[i]["fp16"] < 1e-2  # |d| / |o|: float16 rounding
        # for one segment the first-order RMS is the same with or without the
        # uncorrelated-errors assumption
        assert est.predict_signed(single(i)) == pytest.approx(est.predict(single(i), "loss", relative=False))


def test_predict_formulas():
    """mean_sq, noise, coefficients, predict and predict_signed on known numbers."""
    signed = [{"fp16": [1e-3, -2e-3, 4e-4]}, {"fp16": [3e-3, 2e-3, -4e-4]}]
    grad_sq = [{"fp16": [4.0, 1.0, 0.0]}, {"fp16": [2.0, 3.0, 2.0]}]
    est = SegmentErrors(policies=list(SEGMENT_POLICIES), num_segments=3, device="cpu", reference="none",
                        half_softmax=False, loss=[2.0, 4.0], signed=signed, loss_noise=0.5,
                        grad_sq_batches=grad_sq, grad_noise=100.0, grad_norm_sq=400.0)
    loss_sq = est.mean_sq("loss")
    assert loss_sq[0]["fp16"] == pytest.approx(5e-6) and loss_sq[1]["fp16_none"] == pytest.approx(4e-6)
    assert loss_sq[2]["fp32"] == 0.0
    assert est.mean_sq("grad")[0]["fp16"] == 3.0 and est.mean_sq("grad")[2]["fp16"] == 1.0
    assert est.coefficients("grad")[1]["fp16"] == pytest.approx(0.02)
    pol = ["fp16", "fp16_none", "none"]
    assert est.predict(pol, "grad", relative=False) == pytest.approx(np.sqrt(3.0 + 2.0))
    assert est.predict(pol, "grad") == pytest.approx(np.sqrt(5.0 / 100.0))
    assert est.predict(pol, "loss") == pytest.approx(np.sqrt((5e-6 + 4e-6) / 0.5))
    # with the correlations: per batch (1e-3 - 2e-3)^2 and (3e-3 + 2e-3)^2
    assert est.predict_signed(pol) == pytest.approx(np.sqrt((1e-6 + 25e-6) / 2))


@pytest.mark.parametrize("reference", ["none", "fp32"])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_rerun_reproduces_the_reference_exactly(device, reference):
    """The measured differences are due to precision alone: rerunning a segment in
    float32 from its recorded input reproduces the reference output bit for bit."""
    m, fns = setup(device)
    tr = m.sequence_model
    params = m.parameters()
    with _Preserved(tr, params, True) as state:
        _, grads, xs, outs, out_grads = _reference_pass(tr, fns[0], reference, params, state, 1024.0)
        assert len(grads) == len(params)
        for i in range(LAYERS):
            for policy in ("fp32", "none"):
                np.testing.assert_array_equal(_rerun(tr, i, xs[i], policy), outs[i])
            assert not np.array_equal(_rerun(tr, i, xs[i], "fp16"), outs[i])
            assert np.any(out_grads[i] != 0)


def test_reference_fp32_and_none_agree():
    m, fns = setup(ndl.cpu())
    a = estimate_segment_errors(m.sequence_model, fns, params=m.parameters(), reference="none")
    b = estimate_segment_errors(m.sequence_model, fns, params=m.parameters(), reference="fp32")
    assert a.signed == b.signed and a.loss == b.loss and a.grad_sq_batches == b.grad_sq_batches
    assert a.grad_noise == b.grad_noise


# --------------------------------------------------------------------------- accuracy


@pytest.mark.parametrize("half_softmax", [False, True])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_first_order_matches_single_segment_change(device, half_softmax):
    """s_bi = <grad_{o_i} L_b, o_i^fp16 - o_i^fp32> predicts the actual loss change of
    switching only segment i to float16, sign included, on every batch."""
    m, fns = setup(device, half_softmax=half_softmax)
    est = estimate_segment_errors(m.sequence_model, fns, params=m.parameters(), gradients=False)
    for i in range(LAYERS):
        actual = np.array(measure_precision_error(m.sequence_model, fns, single(i), params=m.parameters(),
                                                  gradients=False)["signed"])
        first_order = np.array([b["fp16"][i] for b in est.signed])
        assert np.all(np.abs(first_order - actual) <= 0.1 * np.abs(actual).max()), (i, first_order, actual)


def test_gradient_coefficients_are_the_single_segment_measurements(cpu_estimate):
    m, fns, est = cpu_estimate
    for i in range(LAYERS):
        r = measure_precision_error(m.sequence_model, fns, single(i), params=m.parameters())
        assert [b["fp16"][i] for b in est.grad_sq_batches] == pytest.approx(r["grad_sq"], rel=1e-9)


def test_sum_of_squares_model(cpu_estimate):
    """Over random configurations the sum of squares predicts the measured RMS
    gradient error closely and orders the configurations like it. The rounding
    errors of different segments are uncorrelated only on average: in a model this
    small their cross terms still move the measurement by up to about 30%. The
    scalar loss error is less regular (correlated across segments)."""
    m, fns, est = cpu_estimate
    rng = np.random.RandomState(3)
    configs = [c for c in [list(rng.choice(["fp32", "fp16", "none", "fp16_none"], LAYERS)) for _ in range(10)]
               + [["fp16"] * LAYERS] if any(p in ("fp16", "fp16_none") for p in c)]
    pg, mg, pl, ml = [], [], [], []
    for c in configs:
        r = measure_precision_error(m.sequence_model, fns, c, params=m.parameters())
        pg.append(est.predict(c, "grad", relative=False))
        mg.append(r["grad_rms"])
        pl.append(est.predict(c, "loss", relative=False))
        ml.append(r["loss_rms"])
    ratio = np.array(mg) / np.array(pg)
    assert np.all((0.3 < ratio) & (ratio < 1.5)), ratio
    assert 0.7 < np.median(ratio) < 1.3
    assert rank_corr(pg, mg) > 0.8
    assert np.all((0.2 < np.array(ml) / np.array(pl)) & (np.array(ml) / np.array(pl) < 3))
    # float32-only configurations are exact
    r = measure_precision_error(m.sequence_model, fns, ["none", "fp32", "none"], params=m.parameters())
    assert r["loss_rms"] == r["grad_rms"] == 0.0


# --------------------------------------------------------------------------- batch noise


def test_batch_noise(cpu_estimate):
    """loss_noise = Var_b(L_b); grad_noise = E_b|g_b - mean g|^2, both with Bessel's
    correction, from the float32 reference of every batch."""
    m, fns, est = cpu_estimate
    tr = m.sequence_model
    assert est.loss_noise == pytest.approx(np.var(est.loss, ddof=1))
    grads = []
    tr.set_policies("fp32")
    tr.eval()
    try:
        for fn in fns:
            for p in m.parameters():
                p.grad = None
            fn().backward()
            grads.append(np.concatenate([p.grad.numpy().astype(np.float64).ravel() for p in m.parameters()]))
    finally:
        tr.train()
        for p in m.parameters():
            p.grad = None
    G = np.stack(grads)
    expected = np.sum((G - G.mean(0)) ** 2) / (len(fns) - 1)
    assert est.grad_noise == pytest.approx(expected, rel=1e-6)
    assert est.grad_norm_sq == pytest.approx(np.mean(np.sum(G ** 2, axis=1)), rel=1e-6)


def test_one_batch_has_no_noise():
    m, fns = setup(ndl.cpu(), batches=1)
    est = estimate_segment_errors(m.sequence_model, fns, params=m.parameters())
    assert est.loss_noise is None and est.grad_noise is None
    assert est.predict("fp16", "grad", relative=False) > 0  # absolute errors still work
    for metric in ("loss", "grad"):
        with pytest.raises(ValueError, match="two batches"):
            est.predict("fp16", metric)
        with pytest.raises(ValueError, match="two batches"):
            est.coefficients(metric)


def test_without_gradients():
    m, fns = setup(ndl.cpu())
    est = estimate_segment_errors(m.sequence_model, fns, params=m.parameters(), gradients=False)
    assert est.grad_sq_batches is None and est.grad_noise is None
    assert est.predict("fp16", "loss") > 0
    with pytest.raises(ValueError, match="gradients=False"):
        est.predict("fp16", "grad", relative=False)
    with pytest.raises(ValueError, match="gradients=True"):
        est.noise("grad")


def test_loss_scale_backs_off_on_overflow(cpu_estimate):
    """A loss scale that overflows the float16 gradients is halved until they are
    finite, as a GradScaler would."""
    m, fns, est = cpu_estimate
    big = estimate_segment_errors(m.sequence_model, fns[:2], params=m.parameters(), loss_scale=2.0 ** 60)
    assert all(np.isfinite(v) for b in big.grad_sq_batches for v in b["fp16"])
    for b in range(2):
        assert big.grad_sq_batches[b]["fp16"] == pytest.approx(est.grad_sq_batches[b]["fp16"], rel=0.1)


# --------------------------------------------------------------------------- measured errors


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_measure_precision_error(device):
    m, fns = setup(device)
    tr = m.sequence_model
    for exact in ("fp32", "none", ["none", "fp32", "none"]):
        r = measure_precision_error(tr, fns, exact, params=m.parameters())
        assert r["loss_rms"] == 0.0 and r["grad_rms"] == 0.0 and r["grad_rel"] == 0.0
    a = measure_precision_error(tr, fns, "fp16", params=m.parameters())
    b = measure_precision_error(tr, fns, "fp16_none", params=m.parameters())
    assert a == b  # bit-identical losses and gradients
    assert 0 < a["loss_rms"] < 1e-2 and 0 < a["grad_rel"] < 0.1
    assert a["grad_rms"] == pytest.approx(np.sqrt(np.mean(a["grad_sq"])))
    c = measure_precision_error(tr, fns, "fp16", params=m.parameters(), gradients=False)
    assert c["signed"] == a["signed"] and c["grad_rms"] is None and c["grad_sq"] is None


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
    assert a == b
    assert calls == [] and tr.segment_hook is user_hook
    assert tr.policies == ["none", "fp16", "fp32"]
    assert all(p.grad is g for p, g in zip(m.parameters(), grads))
    assert tr.training and all(mod.training for mod in tr._children())


def test_half_softmax_recorded(cpu_estimate):
    m, fns, est = cpu_estimate
    tr = m.sequence_model
    tr.half_softmax = True
    try:
        hs = estimate_segment_errors(tr, fns, params=m.parameters(), gradients=False)
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
    with pytest.raises(ValueError, match="metric"):
        est.predict("fp16", "accuracy")


def test_save_load_round_trip(cpu_estimate, tmp_path):
    _, _, est = cpu_estimate
    path = tmp_path / "errors.json"
    est.save(str(path))
    loaded = SegmentErrors.load(str(path))
    assert loaded == est
    for pol in ("fp16", ["fp16", "none", "fp16_none"]):
        for metric in ("loss", "grad"):
            assert loaded.predict(pol, metric) == est.predict(pol, metric)
        assert loaded.predict_signed(pol) == est.predict_signed(pol)
    text = est.summary()
    assert "segment 0" in text and "of noise" in text and "grad RMS" in text
