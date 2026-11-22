"""MILP solver for the segment policies: the fastest configuration that fits a
memory budget and an error limit.

With one binary variable x[i, p] per segment i and policy p (sum_p x[i, p] = 1):

    minimize    sum_{i,p} time_delta[i][p] x[i, p]                (+ base_time)
    subject to  static + correction + const_k + sum_{i,p} coef_k[i][p] x[i, p] <= budget
                                                for every memory phase k
                sum_{i,p} c[i][p] x[i, p] <= eps^2

from a SegmentProfile (time_delta; the memory phases of memory_phases(); for a
profile saved before the phase model, the single additive row base_peak +
sum peak_delta) and a SegmentErrors (c = coefficients(metric): mean squared
precision error / batch noise, so eps is the allowed RMS error as a fraction of
the batch noise's standard deviation). Every constraint is linear, so the model
is a mixed-integer linear program; it is solved exactly with scipy.optimize.milp
(HiGHS, scipy >= 1.9).

The peak of the chosen configuration enters through one continuous variable
M >= every phase. Ties are broken in a second solve: among the configurations at
most `time_tolerance` seconds slower than the fastest, the one with the lowest
peak (tie_break="memory") or the lowest error (tie_break="error"). Margins make
the limits stricter: the memory budget is multiplied by (1 - memory_margin) and
eps divided by error_margin.

memory_frontier() and error_frontier() trace the exact Pareto frontiers (time vs
peak at a fixed eps, time vs error at a fixed budget) with the epsilon-constraint
method: the fastest plan, then repeatedly the fastest plan strictly below the
last plan's peak (or error), until none is feasible.
"""
import contextlib
import itertools
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Plan", "solve", "memory_frontier", "error_frontier"]

MIB = 2.0 ** 20
_MS = 1e3  # the objective is solved in milliseconds, the memory rows in MiB


def _scipy():
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as e:  # pragma: no cover - exercised by a test with scipy hidden
        raise ImportError("the policy solver needs scipy >= 1.9 (scipy.optimize.milp): "
                          "pip install 'scipy>=1.9'") from e
    return milp, LinearConstraint, Bounds


@contextlib.contextmanager
def _quiet_c_stdout():
    """Silence output written to file descriptor 1 by C code (HiGHS occasionally
    prints a debug line from inside its MIP solver)."""
    try:
        sys.stdout.flush()
        fd = sys.stdout.fileno()
    except (AttributeError, ValueError, OSError):  # no real stdout (e.g. captured)
        yield
        return
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


@dataclass
class Plan:
    """A configuration chosen by solve(), with its predicted costs."""

    policies: List[str]
    feasible: bool
    status: str
    time: float  # predicted seconds per step
    peak: int  # predicted peak bytes of a step (including static memory)
    error: Optional[float]  # predicted RMS error / batch noise (None without errors)
    budget: Optional[float] = None  # effective memory budget in bytes (after the margin)
    eps: Optional[float] = None  # effective error limit (after the margin)
    metric: Optional[str] = None
    limited_by: List[str] = field(default_factory=list)  # "memory" / "error": would be faster without it

    def summary(self) -> str:
        head = "plan" if self.feasible else "INFEASIBLE plan"
        line = (f"{head}: {','.join(self.policies)} | time {self.time * 1e3:.2f} ms | "
                f"peak {self.peak / MIB:.1f} MiB")
        if self.budget is not None:
            line += f" (budget {self.budget / MIB:.1f})"
        if self.error is not None:
            line += f" | {self.metric} error {self.error:.3e}"
            if self.eps is not None:
                line += f" (eps {self.eps:.3e})"
        if self.limited_by:
            line += f" | limited by {', '.join(self.limited_by)}"
        if not self.feasible:
            line += f" | {self.status}"
        return line


# --------------------------------------------------------------------------- model


def _policies(profile, errors, policies) -> List[str]:
    allowed = list(profile.policies) if policies is None else list(policies)
    for p in allowed:
        if p not in profile.policies:
            raise ValueError(f"policy {p!r} was not profiled (profiled: {profile.policies})")
        if errors is not None and p not in errors.policies:
            raise ValueError(f"policy {p!r} has no error estimate (estimated: {errors.policies})")
    if not allowed:
        raise ValueError("no policies allowed")
    return allowed


def _check(profile, errors, eps, metric):
    if errors is not None:
        if errors.num_segments != profile.num_segments:
            raise ValueError(f"the profile has {profile.num_segments} segments, the error "
                             f"estimate {errors.num_segments}")
        if None not in (errors.half_softmax, profile.half_softmax) and errors.half_softmax != profile.half_softmax:
            raise ValueError("the profile and the error estimate were made with different "
                             f"half_softmax settings ({profile.half_softmax} / {errors.half_softmax})")
    if eps is not None:
        if errors is None:
            raise ValueError("an error limit eps needs an error estimate (errors=...)")
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")


def _memory_rows(profile) -> List[Tuple[str, float, list]]:
    """The memory phases as (name, const, coef[i][p]), bytes above static, without
    the correction; a profile from before the phase model has one additive row."""
    if getattr(profile, "fixed", None) is None:
        return [("additive", profile.base_peak, profile.peak_delta)]
    return profile.memory_phases()


class _Model:
    """The MILP of one problem: x[i, p] binary (n * P of them) and the peak M (MiB)."""

    def __init__(self, profile, errors, policies, metric):
        self.profile, self.errors, self.metric = profile, errors, metric
        self.n, self.P = profile.num_segments, policies
        self.keys = list(itertools.product(range(self.n), policies))
        nx = len(self.keys)
        self.nx = nx
        self.time = np.array([profile.time_delta[i][p] * _MS for i, p in self.keys])
        fixed = (profile.static + profile.correction) / MIB
        rows = _memory_rows(profile)
        self.phase_names = [name for name, _, _ in rows]
        # phase_k(x) - M <= -(fixed + const_k): M is the peak, in MiB
        self.A_mem = np.zeros((len(rows), nx + 1))
        self.b_mem = np.zeros(len(rows))
        for k, (_, const, coef) in enumerate(rows):
            self.A_mem[k, :nx] = [coef[i][p] / MIB for i, p in self.keys]
            self.A_mem[k, nx] = -1.0
            self.b_mem[k] = -(fixed + const / MIB)
        self.err = None
        if errors is not None:
            coef = errors.coefficients(metric)
            self.err = np.array([coef[i][p] for i, p in self.keys])
        # one policy per segment
        self.A_one = np.zeros((self.n, nx + 1))
        for k, (i, _) in enumerate(self.keys):
            self.A_one[i, k] = 1.0

    def solve(self, budget, eps, objective, max_time=None):
        """Optimal x for `objective` ("time", "memory" or "error"); budget in bytes
        or None, eps (effective) or None; max_time (ms, relative to base_time)
        caps the time. Returns (status, x) with status "optimal" or "infeasible".

        HiGHS checks constraints only up to its (scaled) feasibility tolerance,
        which for a budget of gigabytes is a few hundred bytes. Every solution is
        therefore checked exactly with the profile's and the error estimate's own
        predictions; a solution over a limit tightens that limit (by twice as
        much each time) and is solved again."""
        mem_bound = None if budget is None else np.floor(budget) + 0.5  # peaks are whole bytes
        err_bound = 1.0 - 2e-7
        # first tightening: a little more than the relative tolerance seen (3e-8)
        shrink_mem, shrink_err = (1.0 if budget is None else max(1.0, 1e-7 * budget)), 1e-7
        for _ in range(64):
            status, x = self._milp(mem_bound, eps, err_bound, objective, max_time)
            if status != "optimal":
                return status, None
            policies = self.policies_of(x)
            ok = True
            if budget is not None:
                peak = self.profile.predict_peak(policies)
                if peak > budget:
                    shrink_mem = max(2 * shrink_mem, 2.0 * (peak - budget))
                    mem_bound = np.floor(budget) + 0.5 - shrink_mem
                    ok = False
            if eps is not None and eps > 0:
                error = self.errors.predict(policies, self.metric)
                if error > eps:
                    shrink_err = max(2 * shrink_err, 1 - (eps / error) ** 2)
                    err_bound = (1.0 - 2e-7) * (1 - shrink_err)
                    ok = False
            if ok:
                return status, x
        raise RuntimeError("the MILP solutions keep violating the limits")

    def _milp(self, mem_bound, eps, err_bound, objective, max_time):
        milp, LinearConstraint, Bounds = _scipy()
        nx = self.nx
        c = np.zeros(nx + 1)
        if objective == "time":
            c[:nx] = self.time
        elif objective == "memory":
            c[nx] = 1.0
        else:
            c[:nx] = self.err
        upper = np.ones(nx + 1)
        upper[nx] = np.inf
        if mem_bound is not None:
            upper[nx] = mem_bound / MIB
        cons = [LinearConstraint(self.A_one, 1.0, 1.0),
                LinearConstraint(self.A_mem, -np.inf, self.b_mem)]
        if eps is not None:
            if eps == 0:
                upper[:nx][self.err > 0] = 0.0  # only error-free policies
            else:
                row = np.append(self.err / eps ** 2, 0.0)
                cons.append(LinearConstraint(row[None, :], -np.inf, err_bound))
        if max_time is not None:
            row = np.append(self.time, 0.0)
            cons.append(LinearConstraint(row[None, :], -np.inf, max_time))
        integrality = np.ones(nx + 1)
        integrality[nx] = 0
        bounds = Bounds(np.zeros(nx + 1), upper)
        with _quiet_c_stdout():
            res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
                       options={"mip_rel_gap": 0.0})
        if res.status == 2:
            # HiGHS's presolve was seen to declare feasible models infeasible (a
            # budget a few bytes below a peak, next to the error row; see
            # tests/solver test_presolve_regression): confirm without it
            with _quiet_c_stdout():
                res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
                           options={"mip_rel_gap": 0.0, "presolve": False})
        if res.status == 2:
            return "infeasible", None
        if not res.success:
            raise RuntimeError(f"the MILP solver failed: {res.message}")
        return "optimal", res.x[:nx]

    def policies_of(self, x) -> List[str]:
        out = [None] * self.n
        for k, (i, p) in enumerate(self.keys):
            if x[k] > 0.5:
                out[i] = p
        return out

    def evaluate(self, policies):
        error = self.errors.predict(policies, self.metric) if self.errors is not None else None
        return self.profile.predict_time(policies), int(self.profile.predict_peak(policies)), error


# --------------------------------------------------------------------------- solve


def solve(profile, errors=None, budget: Optional[float] = None, eps: Optional[float] = None,
          metric: str = "grad", memory_margin: float = 0.0, error_margin: float = 1.0,
          policies: Optional[Sequence[str]] = None, time_tolerance: float = 0.0,
          tie_break: str = "memory", explain: bool = True) -> Plan:
    """The fastest configuration of the profiled segments that fits the limits.

    profile: a SegmentProfile; errors: a SegmentErrors (needed for eps).
    budget: memory budget in bytes for the whole step, as SegmentProfile.predict_peak
        counts it (static memory included); None: no memory limit.
    eps: allowed RMS precision error as a fraction of the batch noise
        (SegmentErrors.predict with `metric`); None: no error limit.
    memory_margin / error_margin: the effective limits are budget * (1 - memory_margin)
        and eps / error_margin.
    policies: the policies to choose from (default: every profiled one).
    time_tolerance: seconds; among the configurations at most this much slower
        than the fastest, the one with the lowest peak (tie_break="memory") or
        error (tie_break="error") is returned.
    explain: also find out which limits make the plan slower (Plan.limited_by;
        up to two more solves).

    Predicted peaks are whole bytes and the plan is exact to the byte unless a
    configuration's peak lies within about 2e-7 of the budget (HiGHS's
    feasibility tolerance); such configurations may be excluded.

    If nothing fits, the returned plan has feasible=False and is the configuration
    with the lowest peak under the error limit (or, if even that fails, the one
    with the lowest error).
    """
    if not 0 <= memory_margin < 1:
        raise ValueError(f"memory_margin must be in [0, 1), got {memory_margin}")
    if error_margin < 1:
        raise ValueError(f"error_margin must be >= 1, got {error_margin}")
    if tie_break not in ("memory", "error"):
        raise ValueError(f"tie_break must be 'memory' or 'error', got {tie_break!r}")
    _check(profile, errors, eps, metric)
    allowed = _policies(profile, errors, policies)
    if tie_break == "error" and errors is None:
        raise ValueError("tie_break='error' needs an error estimate")
    model = _Model(profile, errors, allowed, metric)
    budget_eff = None if budget is None else budget * (1 - memory_margin)
    eps_eff = None if eps is None else eps / error_margin

    status, x = model.solve(budget_eff, eps_eff, "time")
    if status == "infeasible":
        return _infeasible(model, budget_eff, eps_eff, metric)
    best = float(model.time @ x)
    # second solve: the tie-break among the (nearly) fastest configurations
    slack = time_tolerance * _MS + 1e-7 * max(1.0, abs(best))
    status, x2 = model.solve(budget_eff, eps_eff, tie_break, max_time=best + slack)
    chosen = model.policies_of(x2 if status == "optimal" else x)  # x is feasible in any case
    time_, peak, error = model.evaluate(chosen)
    assert budget_eff is None or peak <= budget_eff
    assert eps_eff is None or error <= eps_eff

    limited_by = []
    for name, b, e in ((("memory", None, eps_eff), ("error", budget_eff, None)) if explain else ()):
        if (name == "memory" and budget_eff is None) or (name == "error" and eps_eff is None):
            continue
        _, x_free = model.solve(b, e, "time")
        if float(model.time @ x_free) < best - 1e-7 * max(1.0, abs(best)):
            limited_by.append(name)
    return Plan(policies=chosen, feasible=True, status="optimal", time=time_, peak=peak, error=error,
                budget=budget_eff, eps=eps_eff, metric=metric if errors is not None else None,
                limited_by=limited_by)


def _infeasible(model, budget_eff, eps_eff, metric):
    status, x = model.solve(None, eps_eff, "memory")
    if status == "optimal":
        why = "no configuration fits the memory budget; this is the one with the lowest peak"
    else:
        _, x = model.solve(None, None, "error")
        why = "no configuration meets the error limit; this is the one with the lowest error"
    chosen = model.policies_of(x)
    time_, peak, error = model.evaluate(chosen)
    return Plan(policies=chosen, feasible=False, status=why, time=time_, peak=peak, error=error,
                budget=budget_eff, eps=eps_eff, metric=metric if model.errors is not None else None)


# --------------------------------------------------------------------------- frontiers


def memory_frontier(profile, errors=None, eps: Optional[float] = None, metric: str = "grad",
                    policies: Optional[Sequence[str]] = None, max_points: int = 1000) -> List[Plan]:
    """The time / peak Pareto frontier under the error limit eps: plans with
    strictly decreasing peak and non-decreasing time, from the fastest plan to the
    one with the lowest peak; every other configuration is dominated by one of them."""
    plans, budget = [], None
    while len(plans) < max_points:
        plan = solve(profile, errors, budget=budget, eps=eps, metric=metric, policies=policies,
                     explain=False)
        if not plan.feasible:
            break
        plans.append(plan)
        budget = plan.peak - 1  # strictly lower peak next
    return plans


def error_frontier(profile, errors, budget: Optional[float] = None, metric: str = "grad",
                   policies: Optional[Sequence[str]] = None, max_points: int = 1000) -> List[Plan]:
    """The time / error Pareto frontier within the memory budget: plans with
    strictly decreasing error and non-decreasing time, down to the fastest plan
    with the lowest error."""
    plans, eps = [], None
    while len(plans) < max_points:
        plan = solve(profile, errors, budget=budget, eps=eps, metric=metric, policies=policies,
                     tie_break="error", explain=False)
        if not plan.feasible:
            break
        plans.append(plan)
        if plan.error <= 0:
            break
        eps = plan.error * (1 - 1e-6)  # strictly lower error next
    return plans
