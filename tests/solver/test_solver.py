"""needle.policy.solver: the MILP that picks the segment policies."""
import itertools
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

pytest.importorskip("scipy", minversion="1.9")

import needle as ndl
import needle.nn as nn
from needle.nn.nn_transformer import SEGMENT_POLICIES
from needle.policy import (Plan, SegmentErrors, SegmentProfile, error_frontier, estimate_segment_errors,
                           measure_precision_error, measure_step, memory_frontier, profile_segments, solve)
from needle.policy.profiler import FIXED_TERMS

POLICIES = list(SEGMENT_POLICIES)  # fp32, fp16, none, fp16_none
MIB = 2 ** 20


# --------------------------------------------------------------------------- synthetic problems


def make_profile(n, seed, half_softmax=False):
    """A random but plausible profile: fp16 faster and smaller, recomputing slower
    and much smaller, fp16_none in between in time."""
    rng = np.random.RandomState(seed)
    mb = lambda lo, hi: int(rng.uniform(lo, hi) * MIB)
    td, held, fpk, bh, bp, pdelta = [], [], [], [], [], []
    for _ in range(n):
        t16, tre = -rng.uniform(5, 40) * 1e-3, rng.uniform(10, 40) * 1e-3
        td.append({"fp32": 0.0, "fp16": t16, "none": tre, "fp16_none": t16 + rng.uniform(0.3, 0.9) * tre})
        h32 = mb(800, 1400)
        h16 = int(h32 * rng.uniform(0.5, 0.9))
        hb = mb(50, 200)
        held.append({"fp32": h32, "fp16": h16, "none": hb, "fp16_none": int(hb * 0.8)})
        fpk.append({"fp32": h32 + mb(0, 50), "fp16": h16 + mb(0, 50), "none": h32 + mb(0, 50),
                    "fp16_none": h16 + mb(0, 50)})
        g32 = mb(800, 1400)
        bh.append({"fp32": g32, "fp16": int(g32 * rng.uniform(0.8, 1.0)), "none": mb(50, 200),
                   "fp16_none": mb(40, 150)})
        bp.append({"fp32": g32 + mb(0, 50), "fp16": int(g32 * 0.9) + mb(0, 50), "none": h32 + mb(100, 300),
                   "fp16_none": h16 + mb(100, 300)})
        pdelta.append({p: 0 for p in POLICIES})
    fixed = {k: mb(10, 300) for k in FIXED_TERMS}
    prof = SegmentProfile(
        policies=list(POLICIES), num_segments=n, device="cpu", key=None, mma=None, base_time=0.9,
        time_delta=td, time_spread=[{p: 0.0 for p in POLICIES} for _ in range(n)],
        forward_time=[{p: 0.0 for p in POLICIES} for _ in range(n)], held=held, forward_peak=fpk,
        recompute=[{} for _ in range(n)], static=mb(10, 30), base_peak=0, peak_delta=pdelta,
        correction=int(rng.uniform(0, 5) * MIB), half_softmax=half_softmax,
        backward_held=bh, backward_peak=bp, fixed=fixed)
    prof.base_peak = max(prof.peak_phases("fp32").values())
    for i in range(n):  # single-switch peak changes, for the legacy additive model
        for p in POLICIES:
            config = ["fp32"] * n
            config[i] = p
            prof.peak_delta[i][p] = max(prof.peak_phases(config).values()) - prof.base_peak
    return prof


def make_errors(n, seed, half_softmax=False):
    rng = np.random.RandomState(seed + 1000)
    batches = []
    for _ in range(3):
        batches.append({"fp16": list(rng.uniform(0.5, 2.0, n) * np.geomspace(4e-4, 1e-4, n))})
    signed = [{"fp16": list(rng.randn(n) * 1e-4)} for _ in range(3)]
    return SegmentErrors(policies=list(POLICIES), num_segments=n, device="cpu", reference="none",
                         half_softmax=half_softmax, loss=[10.0, 10.5, 9.5], signed=signed, loss_noise=0.25,
                         grad_sq_batches=batches, grad_noise=1.0, grad_norm_sq=4.0)


def brute(profile, errors, budget=None, eps=None, policies=POLICIES, metric="grad"):
    """Every configuration with (time, peak, error, feasible)."""
    out = []
    for c in itertools.product(policies, repeat=profile.num_segments):
        c = list(c)
        t, pk = profile.predict_time(c), profile.predict_peak(c)
        e = errors.predict(c, metric) if errors is not None else None
        ok = (budget is None or pk <= budget) and (eps is None or e <= eps * (1 + 1e-12))
        out.append((t, pk, e, ok, c))
    return out


def limits(profile, errors, rng):
    peaks = [profile.predict_peak(list(c)) for c in itertools.product(POLICIES, repeat=profile.num_segments)]
    budget = float(rng.uniform(min(peaks), max(peaks)))
    eps = float(rng.uniform(0, errors.predict("fp16")))
    return budget, eps


# --------------------------------------------------------------------------- exactness


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force(n, seed):
    """The MILP finds the fastest feasible configuration (and, among the fastest,
    the one with the lowest peak), for random budgets and error limits."""
    prof, err = make_profile(n, seed), make_errors(n, seed)
    rng = np.random.RandomState(seed)
    for _ in range(4):
        budget, eps = limits(prof, err, rng)
        plan = solve(prof, err, budget=budget, eps=eps)
        cands = [r for r in brute(prof, err, budget, eps) if r[3]]
        if not cands:
            assert not plan.feasible
            continue
        best = min(r[0] for r in cands)
        assert plan.feasible and plan.status == "optimal"
        assert plan.time == pytest.approx(best, abs=1e-9)
        ties = [r for r in cands if r[0] <= best + 1e-9]
        assert plan.peak == min(r[1] for r in ties)
        assert plan.peak <= budget and plan.error <= eps
        assert (plan.time, plan.peak, plan.error) == (prof.predict_time(plan.policies),
                                                     prof.predict_peak(plan.policies),
                                                     err.predict(plan.policies))


def test_byte_exact_budgets():
    """HiGHS's tolerance at gigabyte scale is a few hundred bytes; plans are still
    exact to the byte: a budget one byte below a configuration's peak excludes it."""
    prof = make_profile(3, 20)
    rows = brute(prof, None)
    for t, pk, _, _, c in sorted(rows)[:: 7]:
        plan = solve(prof, budget=pk - 1)
        cands = [r for r in rows if r[1] <= pk - 1]
        if not cands:
            assert not plan.feasible
            continue
        assert plan.peak <= pk - 1
        assert plan.time == pytest.approx(min(r[0] for r in cands), abs=1e-9)
        plan = solve(prof, budget=pk)  # exactly at the budget is allowed
        assert plan.peak <= pk and plan.time <= t + 1e-9


def test_presolve_regression():
    """With HiGHS's presolve on, this problem (a budget one byte below the fastest
    plan's peak, with an error limit) was declared infeasible although recomputing
    every segment fits easily."""
    prof, err = make_profile(12, 2), make_errors(12, 2)
    fastest = solve(prof, err, eps=0.01)
    plan = solve(prof, err, budget=fastest.peak - 1, eps=0.01)
    assert plan.feasible and plan.peak < fastest.peak and plan.time >= fastest.time
    assert len(memory_frontier(prof, err, eps=0.01, max_points=4)) == 4


def test_unconstrained_is_the_fastest_configuration():
    prof = make_profile(4, 1)
    plan = solve(prof)
    fastest = min(brute(prof, None), key=lambda r: (r[0], r[1]))
    assert plan.policies == fastest[4] and plan.error is None and plan.limited_by == []
    assert plan.budget is None and plan.eps is None


def test_eps_zero_allows_only_float32_numerics():
    prof, err = make_profile(4, 2), make_errors(4, 2)
    plan = solve(prof, err, eps=0.0)
    assert set(plan.policies) <= {"fp32", "none"} and plan.error == 0.0
    assert plan.time == min(r[0] for r in brute(prof, err, policies=["fp32", "none"]))


def test_policy_subset():
    prof, err = make_profile(3, 3), make_errors(3, 3)
    plan = solve(prof, err, policies=["fp32", "none"])
    assert set(plan.policies) <= {"fp32", "none"}
    plan = solve(prof, err, policies=["fp16", "fp16_none"], budget=1e15)
    assert set(plan.policies) <= {"fp16", "fp16_none"}


def test_more_room_is_never_slower():
    prof, err = make_profile(4, 4), make_errors(4, 4)
    rows = brute(prof, err)
    peaks = sorted({r[1] for r in rows})
    times = [solve(prof, err, budget=b, eps=0.02).time for b in peaks[::8] if solve(prof, err, budget=b, eps=0.02).feasible]
    assert all(a >= b - 1e-12 for a, b in zip(times, times[1:]))
    times = [solve(prof, err, eps=e).time for e in np.linspace(0, err.predict("fp16"), 6)]
    assert all(a >= b - 1e-12 for a, b in zip(times, times[1:]))


def test_time_tolerance_trades_speed_for_memory():
    prof = make_profile(4, 5)
    rows = brute(prof, None)
    best = min(r[0] for r in rows)
    for tol in (0.0, 0.01, 0.05):
        plan = solve(prof, time_tolerance=tol)
        near = [r for r in rows if r[0] <= best + tol + 1e-9]
        assert plan.peak == min(r[1] for r in near) and plan.time <= best + tol + 1e-9


def test_tie_break_on_error():
    prof, err = make_profile(3, 6), make_errors(3, 6)
    rows = brute(prof, err)
    best = min(r[0] for r in rows)
    plan = solve(prof, err, time_tolerance=0.03, tie_break="error")
    near = [r for r in rows if r[0] <= best + 0.03 + 1e-9]
    assert plan.error == pytest.approx(min(r[2] for r in near))


# --------------------------------------------------------------------------- limits, margins, infeasibility


def test_margins_tighten_the_limits():
    prof, err = make_profile(4, 7), make_errors(4, 7)
    budget = prof.predict_peak("fp32") * 0.6
    plain = solve(prof, err, budget=budget, eps=0.015)
    tight = solve(prof, err, budget=budget, eps=0.015, memory_margin=0.1, error_margin=1.5)
    assert tight.budget == pytest.approx(0.9 * budget) and tight.eps == pytest.approx(0.01)
    if tight.feasible:
        assert tight.peak <= 0.9 * budget and tight.error <= 0.01 * (1 + 1e-9)
        assert tight.time >= plain.time - 1e-12
    with pytest.raises(ValueError, match="memory_margin"):
        solve(prof, err, budget=budget, memory_margin=1.0)
    with pytest.raises(ValueError, match="error_margin"):
        solve(prof, err, eps=0.01, error_margin=0.5)


def test_limited_by():
    prof, err = make_profile(4, 8), make_errors(4, 8)
    free = solve(prof, err)
    assert solve(prof, err, budget=free.peak, eps=free.error * 1.01).limited_by == []
    lowest = min(r[1] for r in brute(prof, err))
    assert "memory" in solve(prof, err, budget=lowest * 1.02).limited_by
    assert solve(prof, err, eps=0.0).limited_by == ["error"]


def test_infeasible_budget_returns_the_lowest_peak():
    prof, err = make_profile(3, 9), make_errors(3, 9)
    rows = brute(prof, err, eps=0.01)
    lowest = min(r[1] for r in rows if r[3])
    plan = solve(prof, err, budget=lowest - 1, eps=0.01)
    assert not plan.feasible and "memory budget" in plan.status
    assert plan.peak == lowest and plan.error <= 0.01 * (1 + 1e-9)
    assert "INFEASIBLE" in plan.summary()


def test_infeasible_error_limit_returns_the_lowest_error():
    prof, err = make_profile(3, 10), make_errors(3, 10)
    plan = solve(prof, err, eps=1e-6, policies=["fp16", "fp16_none"])  # no error-free policy allowed
    assert not plan.feasible and "error limit" in plan.status


def test_argument_checks():
    prof, err = make_profile(3, 11), make_errors(3, 11)
    with pytest.raises(ValueError, match="error estimate"):
        solve(prof, eps=0.01)
    with pytest.raises(ValueError, match="segments"):
        solve(prof, make_errors(4, 11))
    with pytest.raises(ValueError, match="half_softmax"):
        solve(prof, make_errors(3, 11, half_softmax=True))
    with pytest.raises(ValueError, match="not profiled"):
        solve(prof, err, policies=["fp32", "bf16"])
    with pytest.raises(ValueError, match="eps"):
        solve(prof, err, eps=-1.0)
    with pytest.raises(ValueError, match="tie_break"):
        solve(prof, err, tie_break="time")
    with pytest.raises(ValueError, match="metric"):
        solve(prof, err, eps=0.01, metric="accuracy")


def test_loss_metric():
    prof, err = make_profile(3, 12), make_errors(3, 12)
    eps = err.predict("fp16", "loss") / 2
    plan = solve(prof, err, eps=eps, metric="loss")
    assert plan.metric == "loss" and plan.error <= eps * (1 + 1e-9)
    cands = [r for r in brute(prof, err, eps=eps, metric="loss") if r[3]]
    assert plan.time == pytest.approx(min(r[0] for r in cands), abs=1e-9)


def test_legacy_profile_uses_the_additive_model():
    """A profile saved before the phase model: one additive memory row."""
    d = make_profile(3, 13).to_dict()
    for k in ("backward_held", "backward_peak", "fixed", "half_softmax"):
        del d[k]
    legacy = SegmentProfile.from_dict(d)
    budget = legacy.predict_peak("none") * 1.3
    plan = solve(legacy, budget=budget)
    cands = [r for r in brute(legacy, None, budget) if r[3]]
    assert plan.time == pytest.approx(min(r[0] for r in cands), abs=1e-9) and plan.peak <= budget


def test_scipy_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    with pytest.raises(ImportError, match="pip install"):
        solve(make_profile(2, 0))


# --------------------------------------------------------------------------- frontiers


def dominated(row, frontier, key):
    return any(p.time <= row[0] + 1e-9 and key(p) <= key(row) for p in frontier)


def test_memory_frontier_is_the_pareto_frontier():
    prof, err = make_profile(4, 14), make_errors(4, 14)
    eps = 0.015
    front = memory_frontier(prof, err, eps=eps)
    assert len(front) >= 2
    peaks, times = [p.peak for p in front], [p.time for p in front]
    assert all(a > b for a, b in zip(peaks, peaks[1:])) and all(a <= b + 1e-12 for a, b in zip(times, times[1:]))
    rows = [r for r in brute(prof, err, eps=eps) if r[3]]
    assert all(any(p.time <= r[0] + 1e-9 and p.peak <= r[1] for p in front) for r in rows)
    assert front[-1].peak == min(r[1] for r in rows)  # down to the lowest feasible peak


def test_error_frontier_is_the_pareto_frontier():
    prof, err = make_profile(4, 15), make_errors(4, 15)
    budget = prof.predict_peak("fp32") * 0.7
    front = error_frontier(prof, err, budget=budget)
    errs, times = [p.error for p in front], [p.time for p in front]
    assert all(a > b for a, b in zip(errs, errs[1:])) and all(a <= b + 1e-12 for a, b in zip(times, times[1:]))
    rows = [r for r in brute(prof, err, budget) if r[3]]
    assert all(any(p.time <= r[0] + 1e-9 and p.error <= r[2] * (1 + 1e-9) for p in front) for r in rows)
    assert front[-1].error == min(r[2] for r in rows)


# --------------------------------------------------------------------------- end to end


def test_end_to_end_on_cpu():
    """Profile and estimate a small model, solve for 60% of the fp32 peak and an
    error limit, and measure the plan: the peak is as predicted (needle's counter
    is deterministic) and within the budget, and the error is within the limit."""
    from models import LanguageModel

    dev = ndl.cpu()
    np.random.seed(0)
    m = LanguageModel(32, 60, 64, num_layers=3, seq_model="transformer", device=dev, policies="fp32")
    fns = []
    for b in range(3):
        rng = np.random.RandomState(10 + b)
        x = ndl.Tensor(rng.randint(0, 60, (12, 4)), device=dev, requires_grad=False)
        y = ndl.Tensor(rng.randint(0, 60, 48), device=dev, requires_grad=False)
        fns.append(lambda x=x, y=y: nn.SoftmaxLoss()(m(x)[0], y))
    tr = m.sequence_model
    prof = profile_segments(tr, fns[0], params=m.parameters(), warmup=0, repeats=1)
    err = estimate_segment_errors(tr, fns, params=m.parameters())
    budget = 0.6 * prof.predict_peak("fp32")
    eps = 0.6 * err.predict("fp16")
    plan = solve(prof, err, budget=budget, eps=eps)
    assert plan.feasible and isinstance(plan, Plan)
    measured = measure_step(tr, fns[0], plan.policies, params=m.parameters(), warmup=0, repeats=1)
    assert measured["peak"] + measured["static"] == plan.peak <= budget
    actual = measure_precision_error(tr, fns, plan.policies, params=m.parameters())
    assert actual["grad_rms"] / np.sqrt(err.noise("grad")) <= 1.5 * eps
    assert "plan:" in plan.summary()


def test_plan_policies_script(tmp_path, capsys):
    sys.path.append(".")
    from apps.plan_policies import main

    prof, err = make_profile(4, 21), make_errors(4, 21)
    prof.save(str(tmp_path / "prof.json"))
    err.save(str(tmp_path / "err.json"))
    common = ["--profile", str(tmp_path / "prof.json"), "--errors", str(tmp_path / "err.json")]
    plan = main(common + ["--budget-fraction", "0.5", "--eps", "0.01"])
    out = capsys.readouterr().out
    assert plan == solve(prof, err, budget=0.5 * prof.predict_peak("fp32"), eps=0.01)
    assert "plan:" in out and "uniform configurations" in out
    front = main(common + ["--frontier", "memory", "--eps", "0.02"])
    assert front == memory_frontier(prof, err, eps=0.02) and "Pareto frontier" in capsys.readouterr().out
    front = main(common + ["--frontier", "error", "--budget-mib", str(prof.predict_peak("fp32") * 0.7 / MIB)])
    assert len(front) >= 1 and "error Pareto frontier" in capsys.readouterr().out


def test_model_rows_reproduce_the_predictions():
    """The MILP is the profile's and the error estimate's own model: for any
    configuration its time, memory and error rows give exactly predict_time,
    predict_peak and the squared relative error."""
    from needle.policy.solver import MIB as S_MIB, _Model

    prof, err = make_profile(4, 22), make_errors(4, 22)
    model = _Model(prof, err, POLICIES, "grad")
    rng = np.random.RandomState(0)
    for _ in range(20):
        c = list(rng.choice(POLICIES, 4))
        x = np.array([1.0 if c[i] == p else 0.0 for i, p in model.keys])
        assert prof.base_time + model.time @ x / 1e3 == pytest.approx(prof.predict_time(c), abs=1e-12)
        peak = (model.A_mem[:, :model.nx] @ x - model.b_mem).max() * S_MIB
        assert peak == pytest.approx(prof.predict_peak(c), abs=1e-3)
        assert model.err @ x == pytest.approx(err.predict(c) ** 2, rel=1e-12)


def test_presolve_fallback():
    """HiGHS's presolve declared this model infeasible (all-'none' fits easily); the
    solver confirms 'infeasible' answers without presolve."""
    from needle.policy.solver import _Model

    prof, err = make_profile(12, 2), make_errors(12, 2)
    model = _Model(prof, err, POLICIES, "grad")
    fastest = solve(prof, err, eps=0.01)
    # the raw MILP may return a configuration within HiGHS's tolerance over the
    # bound (model.solve's exact check removes those); it must not say "infeasible"
    status, x = model._milp(fastest.peak - 2.5, 0.01, 1 - 2e-7, "time", None)
    assert status == "optimal"


def test_error_frontier_breaks_time_ties():
    """Two segments with the same times but different errors: the error frontier
    takes the lower-error one at each time, so no point dominates another."""
    prof, err = make_profile(3, 23), make_errors(3, 23)
    prof.time_delta[1] = dict(prof.time_delta[0])
    for b in err.grad_sq_batches:
        b["fp16"][1] = 3 * b["fp16"][0]
    front = error_frontier(prof, err)
    for a in front:
        for b in front:
            if a is not b:
                assert not (a.time <= b.time + 1e-12 and a.error <= b.error and
                            (a.time < b.time - 1e-12 or a.error < b.error)), (a.summary(), b.summary())


def test_plan_policies_defaults_to_the_stats_files(capsys):
    """Without --profile / --errors the script reads stats/prof_act_hs.json and
    stats/err_act_hs.json (the committed GPU measurements; this also checks that
    they still load and solve with the current code)."""
    import os

    sys.path.append(".")
    from apps.plan_policies import DEFAULT_ERRORS, DEFAULT_PROFILE, main

    if not (os.path.exists(DEFAULT_PROFILE) and os.path.exists(DEFAULT_ERRORS)):
        pytest.skip("no stats files")
    prof = SegmentProfile.load(DEFAULT_PROFILE)
    plan = main(["--budget-fraction", "0.4", "--eps", "0.01"])
    out = capsys.readouterr().out
    assert f"errors: {DEFAULT_ERRORS}" in out and plan.feasible
    assert plan.peak <= 0.4 * prof.predict_peak("fp32") and plan.error <= 0.01
    plan = main(["--no-errors", "--budget-fraction", "0.4"])
    assert plan.error is None and "errors: none" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--no-errors", "--eps", "0.01"])  # an error limit needs the estimate
    with pytest.raises(SystemExit):
        main(["--profile", "stats/no_such_profile.json"])
