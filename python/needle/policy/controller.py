"""Dynamic segment policies during training: re-plan every k epochs.

    opt = ndl.optim.Adam(model.parameters(), lr=1e-3)
    controller = PolicyController(model, budget_fraction=0.4, eps=0.01, replan_every=5,
                                  optimizer=opt, log_path="stats/train_run.json")
    train_ptb(model, data, seq_len=256, n_epochs=20, controller=controller, ...)

or, in a training loop of your own:

    for epoch in range(n_epochs):
        controller.before_epoch(epoch, lambda: sample_ptb_batches(data, seq_len, 8))
        ...   # train one epoch, with controller.scaler as the GradScaler
        controller.after_epoch(epoch, loss=avg_loss, seconds=elapsed, peak=peak_bytes)

At epoch 0 and every `replan_every` epochs after it, before_epoch re-plans:

1. profile: the segments' time and memory costs (needle.policy.profile_segments) on
   the first sampled batch, once per batch shape (kept in a ProfileCache), or a
   profile given to the controller (e.g. loaded from stats/);
2. errors: the precision errors on the sampled training batches
   (estimate_segment_errors; the gradient errors on `grad_batches` of them) -- they
   change as the model trains, which is why the plan is updated;
3. solve: the fastest configuration within the memory budget and the error limit
   eps (needle.policy.solve);
4. apply: transformer.set_policies(plan.policies).

The memory budget counts what the profile counts (needle's NDArray buffers) plus
the optimizer state, which is created at the first optimizer step and is not in
a profile made before training (Adam keeps two copies of the parameters, needle's
SGD one). The profile covers loss and backward; the training loop also holds
memory it cannot see (the optimizer step's temporaries, which run while the
step's graph is still alive, the batch tensors). after_epoch therefore compares
the measured peak of every epoch with the prediction: the largest excess is
added to the static memory at the next re-plan, and an epoch over the budget
makes the next epoch re-plan even if it is not due (unless the plan was already
infeasible: then it is the configuration with the lowest peak). One GradScaler (controller.scaler) is kept for the whole training; it
is enabled when the allowed policies include float16 ones. Re-planning leaves the
numpy random state, the parameters' gradients and the training mode as they were,
so the training itself is the same as with the chosen policies set by hand.

Every re-plan and every epoch is recorded in controller.history / controller.epochs
and, with log_path, written to a JSON file after each of them.
"""
import dataclasses
import json
import os
import time
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .. import nn
from ..amp import GradScaler
from ..backend_ndarray import ndarray as nd
from ..nn.nn_transformer import FP16_POLICIES, SEGMENT_POLICIES
from .error import estimate_segment_errors
from .profiler import ProfileCache, SegmentProfile, profile_segments
from .solver import Plan, solve

__all__ = ["PolicyController", "sample_ptb_batches", "optimizer_state_bytes"]

MIB = 2 ** 20


def sample_ptb_batches(data, seq_len: int, n: int, device=None, dtype: str = "float32"):
    """n full-length (seq_len) batches (x, y) of batchified data (see
    needle.data.datasets.batchify), evenly spaced over the epoch; fewer if the
    data has fewer full-length batches."""
    from ..data.datasets import get_batch

    starts = list(range(0, data.shape[0] - 1, seq_len))
    full = [i for i in starts if i + seq_len <= data.shape[0] - 1]
    if not full:
        raise ValueError(f"the data has no batch of {seq_len} steps")
    picks = sorted(set(int(round(k)) for k in np.linspace(0, len(full) - 1, min(n, len(full)))))
    return [get_batch(data, full[k], seq_len, device=device, dtype=dtype) for k in picks]


def _param_bytes(params) -> int:
    total = 0
    for p in params:
        itemsize = 2 if p.dtype in ("float16", "bfloat16") else 4
        total += int(np.prod(p.shape)) * itemsize
    return total


def optimizer_state_bytes(optimizer, expected: bool = True) -> int:
    """Bytes of the optimizer's state: what it will hold once it has stepped
    (expected=True: Adam two copies of every parameter, needle's SGD one) or what
    it holds now (expected=False)."""
    if optimizer is None:
        return 0
    stores = [getattr(optimizer, name) for name in ("m", "v", "u") if isinstance(getattr(optimizer, name, None), dict)]
    if not expected:
        total = 0
        for store in stores:
            for value in store.values():
                if hasattr(value, "shape"):
                    total += _param_bytes([value])
        return total
    return len(stores) * _param_bytes(optimizer.params)


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


class PolicyController:
    """Chooses the segment policies of a Transformer model during training.

    model: the model to train; its `sequence_model` (or the model itself) must be an
        nn.Transformer. Batches are (x, y) pairs; the loss of a batch is
        loss_fn(model(x)[0], y) (the first element if the model returns a tuple).
    budget / budget_fraction: memory budget of a training step in bytes (needle's
        counter, static memory and optimizer state included), or as a fraction of
        the all-fp32 step's predicted peak; neither: no memory limit.
    eps, metric: error limit (RMS precision error / batch noise, see
        needle.policy.error) and its metric ("grad" or "loss"); None: no limit.
    replan_every: re-plan at epochs 0, k, 2k, ...
    error_batches, grad_batches: training batches for the error estimate (the
        batch noise needs at least 2), and how many of them measure gradient errors.
    memory_margin, error_margin, policies, time_tolerance: as in needle.policy.solve.
    optimizer: the training optimizer, to count its state in the budget.
    profile: a SegmentProfile to use instead of profiling (e.g. loaded from stats/). It
        must come from the same model (sizes and vocabulary) and batch shape; the
        shape is checked against its key. profile_cache: a ProfileCache to share
        profiles between controllers.
    profile_warmup, profile_repeats, mma: passed to profile_segments.
    static_policies: train with these policies (one per segment, or one for all)
        and only log; nothing is profiled, estimated or solved (for baselines).
    log_path: JSON file written after every re-plan and epoch.
    """

    def __init__(self, model, *, budget: Optional[float] = None, budget_fraction: Optional[float] = None,
                 eps: Optional[float] = None, metric: str = "grad", replan_every: int = 5,
                 error_batches: int = 8, grad_batches: Optional[int] = 2,
                 memory_margin: float = 0.0, error_margin: float = 1.0,
                 policies: Optional[Sequence[str]] = None, time_tolerance: float = 0.0,
                 optimizer=None, loss_fn=None, profile: Optional[SegmentProfile] = None,
                 profile_cache: Optional[ProfileCache] = None, profile_warmup: int = 1,
                 profile_repeats: int = 3, mma: Optional[bool] = True,
                 static_policies: Optional[Union[str, Sequence[str]]] = None,
                 init_scale: float = 2.0 ** 16, log_path: Optional[str] = None):
        if budget is not None and budget_fraction is not None:
            raise ValueError("give budget or budget_fraction, not both")
        if budget_fraction is not None and not budget_fraction > 0:
            raise ValueError(f"budget_fraction must be positive, got {budget_fraction}")
        if replan_every < 1:
            raise ValueError(f"replan_every must be at least 1, got {replan_every}")
        if eps is not None and error_batches < 2:
            raise ValueError("an error limit needs at least 2 error_batches (for the batch noise)")
        self.model = model
        self.transformer = getattr(model, "sequence_model", model)
        if not hasattr(self.transformer, "set_policies") or not hasattr(self.transformer, "num_layers"):
            raise ValueError("the model's sequence_model (or the model) must be an nn.Transformer")
        self.params = list(model.parameters())
        self.budget, self.budget_fraction = budget, budget_fraction
        self.eps, self.metric = eps, metric
        self.replan_every = int(replan_every)
        self.error_batches, self.grad_batches = int(error_batches), grad_batches
        self.memory_margin, self.error_margin = memory_margin, error_margin
        self.policies = None if policies is None else list(policies)
        self.time_tolerance = time_tolerance
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else nn.SoftmaxLoss()
        self.fixed_profile = profile
        self.cache = profile_cache if profile_cache is not None else ProfileCache()
        self.profile_warmup, self.profile_repeats, self.mma = profile_warmup, profile_repeats, mma
        self.log_path = log_path
        n = self.transformer.num_layers
        if static_policies is not None:
            static = [static_policies] * n if isinstance(static_policies, str) else list(static_policies)
            self.transformer.set_policies(static)  # validates them
            self.static_policies = list(self.transformer.policies)
        else:
            self.static_policies = None
            if self.transformer.policies is None:
                self.transformer.set_policies("fp32")
        allowed = self.static_policies or self.policies or list(SEGMENT_POLICIES)
        uses_fp16 = any(p in FP16_POLICIES for p in allowed) and (eps is None or eps > 0)
        self.scaler = GradScaler(init_scale=init_scale, enabled=uses_fp16)
        self._state_at_profile = {}  # profile key -> optimizer state bytes held when profiled
        self.unmodeled = 0  # largest measured epoch peak above the prediction (bytes)
        self._over_budget = False  # the last epoch went over the budget: re-plan next
        self.plan: Optional[Plan] = None
        self.profile: Optional[SegmentProfile] = None
        self.errors = None
        self.history: List[dict] = []
        self.epochs: List[dict] = []
        self.results: dict = {}  # anything else to log (e.g. the final test loss)

    # ------------------------------------------------------------------ planning
    def loss_closure(self, x, y) -> Callable:
        return lambda: self.loss_fn(_logits(self.model(x)), y)

    def should_replan(self, epoch: int) -> Optional[str]:
        """Why `epoch` re-plans ("schedule" or "over budget"), or None."""
        if self.static_policies is not None:
            return None
        if epoch % self.replan_every == 0:
            return "schedule"
        if self._over_budget:
            return "over budget"
        return None

    def before_epoch(self, epoch: int, batches) -> Optional[Plan]:
        """Re-plan if `epoch` is a multiple of replan_every (or the last epoch went
        over the budget). `batches`: a list of (x, y) training batches, or a
        callable returning one (only called when re-planning)."""
        reason = self.should_replan(epoch)
        if reason is None:
            return None
        return self.replan(batches() if callable(batches) else batches, epoch, reason=reason)

    def _profile(self, x, y):
        key = f"{x.shape[0]}x{x.shape[1]}"
        if self.fixed_profile is not None:
            if self.fixed_profile.key is not None and self.fixed_profile.key != key:
                raise ValueError(f"the given profile is for batches {self.fixed_profile.key} (seq_len x batch), "
                                 f"the training batches are {key}")
            return self.fixed_profile, False
        if key in self.cache:
            return self.cache.get(key), False
        profile = profile_segments(self.transformer, self.loss_closure(x, y), params=self.params,
                                   warmup=self.profile_warmup, repeats=self.profile_repeats,
                                   mma=self.mma, key=key)
        self.cache.put(key, profile)
        self._state_at_profile[key] = optimizer_state_bytes(self.optimizer, expected=False)
        return profile, True

    def _with_optimizer_state(self, profile):
        """The profile with the optimizer state the training will hold (minus what
        it already held when profiled) and the measured unmodeled memory added to
        its static memory."""
        held = self._state_at_profile.get(profile.key, 0) if self.fixed_profile is None else 0
        reserve = max(0, optimizer_state_bytes(self.optimizer, expected=True) - held)
        static = profile.static + reserve + self.unmodeled
        return dataclasses.replace(profile, static=static), reserve

    def replan(self, batches, epoch: Optional[int] = None, reason: str = "requested") -> Plan:
        """Profile (if needed), estimate the errors on `batches`, solve and apply."""
        if self.static_policies is not None:
            raise RuntimeError("a controller with static_policies does not plan")
        batches = list(batches)
        if not batches:
            raise ValueError("no batches to plan with")
        rng = np.random.get_state()  # profiling and estimating must not change training
        try:
            start = time.perf_counter()
            profile, profiled = self._profile(*batches[0])
            t_profile = time.perf_counter() - start
            errors, t_errors = None, 0.0
            if self.eps is not None:
                start = time.perf_counter()
                fns = [self.loss_closure(x, y) for x, y in batches[:self.error_batches]]
                errors = estimate_segment_errors(self.transformer, fns, params=self.params,
                                                 gradients=self.metric == "grad",
                                                 grad_batches=self.grad_batches)
                t_errors = time.perf_counter() - start
            effective, reserve = self._with_optimizer_state(profile)
            budget = self.budget
            if self.budget_fraction is not None:
                budget = self.budget_fraction * effective.predict_peak("fp32")
            start = time.perf_counter()
            plan = solve(effective, errors, budget=budget, eps=self.eps, metric=self.metric,
                         memory_margin=self.memory_margin, error_margin=self.error_margin,
                         policies=self.policies, time_tolerance=self.time_tolerance)
            t_solve = time.perf_counter() - start
        finally:
            np.random.set_state(rng)
        self.transformer.set_policies(plan.policies)
        self.plan, self.profile, self.errors = plan, profile, errors
        self._over_budget = False
        entry = {
            "epoch": epoch, "reason": reason, "policies": plan.policies, "feasible": plan.feasible,
            "status": plan.status,
            "predicted_time": plan.time, "predicted_peak": plan.peak, "predicted_error": plan.error,
            "limited_by": plan.limited_by, "budget": plan.budget, "eps": plan.eps,
            "optimizer_state": reserve, "unmodeled": self.unmodeled, "profiled": profiled,
            "profile_key": profile.key,
            "seconds": {"profile": t_profile, "errors": t_errors, "solve": t_solve},
        }
        if errors is not None:
            entry["segment_errors"] = [float(np.sqrt(c["fp16"])) for c in errors.coefficients(self.metric)]
            entry["noise"] = errors.noise(self.metric)
            entry["mean_loss"] = float(np.mean(errors.loss))
        self.history.append(entry)
        self.save_log()
        return plan

    # ------------------------------------------------------------------ logging
    def after_epoch(self, epoch: int, loss: Optional[float] = None, acc: Optional[float] = None,
                    seconds: Optional[float] = None, peak: Optional[int] = None,
                    steps: Optional[int] = None) -> dict:
        """Record an epoch: its loss / accuracy, time, measured peak (bytes of
        needle's counter) and number of steps, next to the plan's predictions. A
        peak above the prediction is remembered (see the module docstring)."""
        over = False
        if peak is not None and self.plan is not None:
            self.unmodeled = max(self.unmodeled, int(peak) - int(self.plan.peak))
            over = self.plan.budget is not None and peak > self.plan.budget
            # a feasible plan over the budget: memory the profile did not see,
            # re-plan with it; an infeasible plan is already the lowest-memory one
            self._over_budget = self._over_budget or (over and self.plan.feasible)
        entry = {
            "epoch": epoch, "policies": list(self.transformer.policies),
            "loss": None if loss is None else float(loss), "acc": None if acc is None else float(acc),
            "seconds": seconds, "steps": steps, "measured_peak": peak,
            "predicted_peak": None if self.plan is None else self.plan.peak,
            "predicted_step_time": None if self.plan is None else self.plan.time,
            "over_budget": over, "loss_scale": self.scaler.get_scale(),
            "skipped_steps": self.scaler.skipped_steps,
        }
        self.epochs.append(entry)
        self.save_log()
        return entry

    def log(self) -> dict:
        return {
            "config": {
                "budget": self.budget, "budget_fraction": self.budget_fraction, "eps": self.eps,
                "metric": self.metric, "replan_every": self.replan_every,
                "error_batches": self.error_batches, "grad_batches": self.grad_batches,
                "memory_margin": self.memory_margin, "error_margin": self.error_margin,
                "policies": self.policies, "static_policies": self.static_policies,
                "num_segments": self.transformer.num_layers,
                "half_softmax": getattr(self.transformer, "half_softmax", None),
                "optimizer": None if self.optimizer is None else type(self.optimizer).__name__,
            },
            "replans": self.history,
            "epochs": self.epochs,
            "results": self.results,
        }

    def save_log(self, path: Optional[str] = None) -> None:
        path = path or self.log_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.log(), f, indent=1)

    def summary(self) -> str:
        lines = []
        for h in self.history:
            line = (f"epoch {h['epoch']} ({h['reason']}): {','.join(h['policies'])} | predicted {h['predicted_time'] * 1e3:.1f} ms/step, "
                    f"peak {h['predicted_peak'] / MIB:.1f} MiB")
            if h["predicted_error"] is not None:
                line += f", error {h['predicted_error']:.3e}"
            if not h["feasible"]:
                line += " | INFEASIBLE"
            secs = h["seconds"]
            line += f" | profile {secs['profile']:.1f} s, errors {secs['errors']:.1f} s, solve {secs['solve']:.2f} s"
            lines.append(line)
        return "\n".join(lines)
