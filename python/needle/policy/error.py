"""First-order estimate of the loss error that half precision causes, per segment.

Running segment i (one TransformerLayer) in float16 instead of float32 changes its
output by d_i = o_i^fp16 - o_i^fp32 (both computed from the same float32 input),
and, to first order, the loss by

    s_i = <grad_{o_i} L, d_i>          (Taylor expansion of L around o_i^fp32)

The gradient is that of the float32 reference: one step with every segment at
`reference` ("none" by default: float32 numerics, bit-identical gradients to
"fp32", least memory), whose backward leaves grad_{o_i} L on every segment
boundary. Each segment is then rerun alone in float16 from its recorded float32
input (forward only, no graph), so one estimate costs one step plus one forward.

Per policy p and segment i the error coefficient is

    error[i][p] = mean over batches of |s_i|    for "fp16" and "fp16_none"
                                                  (bit-identical outputs)
    error[i][p] = 0                              for "fp32" and "none"

and the error of a configuration is sum_i error[i][p_i] (predict()): linear in the
per-segment choices, as the optimizer needs, and (to first order) an upper bound
of the loss change, since |sum_i s_i| <= sum_i |s_i|. predict_signed() gives the
first-order estimate |sum_i s_i| itself. Upstream half-precision segments also
change the inputs of later ones; those cross terms are second order.

The estimate runs with the transformer in eval mode (dropout off), so the
reference and the reruns see the same function; everything it changes (policies,
segment hook, training flags, the parameters' .grad) is restored. It depends on
the model's half_softmax setting, which it records. measure_precision_error()
measures the actual loss and gradient error of any configuration, to validate it.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..autograd import no_grad
from ..nn.nn_transformer import SEGMENT_POLICIES, run_segment
from .profiler import _set_policies

__all__ = ["SegmentErrors", "estimate_segment_errors", "measure_precision_error"]

# the precision each policy's segment computes in, and so the rerun that gives
# its perturbation; policies without an entry compute in float32 (error 0)
HALF_PRECISION = {"fp16": "fp16", "fp16_none": "fp16"}

LossFns = Union[Callable, Sequence[Callable]]


@dataclass
class SegmentErrors:
    """First-order loss errors of the segment policies, per segment."""

    policies: List[str]
    num_segments: int
    device: str
    reference: str
    half_softmax: Optional[bool]
    loss: List[float]  # [batch] reference loss
    signed: List[Dict[str, List[float]]]  # [batch] precision -> [segment] s_i
    error: List[Dict[str, float]]  # [segment][policy] mean |s_i| over batches
    perturbation: List[Dict[str, float]] = field(default_factory=list)  # [segment][precision] |d_i| / |o_i|

    def _check(self, policies) -> List[str]:
        policies = [policies] * self.num_segments if isinstance(policies, str) else list(policies)
        if len(policies) != self.num_segments:
            raise ValueError(f"expected {self.num_segments} policies, got {len(policies)}")
        for p in policies:
            if p not in self.policies:
                raise ValueError(f"unknown policy {p!r} (estimated: {self.policies})")
        return policies

    @property
    def mean_loss(self) -> float:
        return float(np.mean(np.abs(self.loss)))

    def predict(self, policies, relative: bool = False) -> float:
        """sum_i error[i][p_i]: the modelled loss error of `policies` (an upper bound
        to first order); relative=True divides by the mean reference loss."""
        policies = self._check(policies)
        total = sum(self.error[i][p] for i, p in enumerate(policies))
        return total / self.mean_loss if relative else total

    def predict_signed(self, policies) -> float:
        """First-order estimate of the loss change itself: the mean over batches of
        |sum_i s_i| over the half-precision segments (not linear in the policies;
        cancellations between segments are kept)."""
        policies = self._check(policies)
        totals = []
        for batch in self.signed:
            totals.append(abs(sum(batch[HALF_PRECISION[p]][i] for i, p in enumerate(policies)
                                  if p in HALF_PRECISION)))
        return float(np.mean(totals))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentErrors":
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "SegmentErrors":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        lines = [f"segment errors on {self.device} (reference={self.reference}, half_softmax="
                 f"{self.half_softmax}, {len(self.loss)} batches, mean loss {self.mean_loss:.5g})"]
        for i in range(self.num_segments):
            cols = [f"{p}: {self.error[i][p]:.3e} ({self.error[i][p] / self.mean_loss:.2e} rel)"
                    for p in self.policies if p in HALF_PRECISION]
            pert = ", ".join(f"|d|/|o| {prec} {v:.2e}" for prec, v in self.perturbation[i].items())
            lines.append(f"  segment {i}: " + " | ".join(cols) + (f" | {pert}" if pert else ""))
        return "\n".join(lines)


class _Preserved:
    """Saves and restores what estimating changes: the transformer's policies,
    segment hook and training flags, and the parameters' .grad. Puts the
    transformer in eval mode (dropout off) if eval_mode."""

    def __init__(self, transformer, params, eval_mode):
        self.transformer = transformer
        self.params = list(params)
        self.eval_mode = eval_mode

    def __enter__(self):
        tr = self.transformer
        self.policies = None if tr.policies is None else list(tr.policies)
        self.hook = tr.segment_hook
        tr.segment_hook = None
        self.grads = [getattr(p, "grad", None) for p in self.params]
        self.modules = [tr] + tr._children()
        self.training = [m.training for m in self.modules]
        if self.eval_mode:
            tr.eval()
        return self

    def __exit__(self, *exc):
        tr = self.transformer
        tr.set_policies(self.policies)
        tr.segment_hook = self.hook
        for p, g in zip(self.params, self.grads):
            p.grad = g
        for m, flag in zip(self.modules, self.training):
            m.training = flag
        return False

    def clear_grads(self):
        for p in self.params:
            p.grad = None


def _as_list(loss_fn: LossFns) -> List[Callable]:
    fns = [loss_fn] if callable(loss_fn) else list(loss_fn)
    if not fns:
        raise ValueError("no loss_fn given")
    return fns


def _reference_pass(transformer, loss_fn, reference, state):
    """One step at `reference`; returns (loss, [segment inputs, detached],
    [segment outputs as float32 numpy], [grad_{o_i} L as float32 numpy])."""
    n = transformer.num_layers
    inputs, outputs = [None] * n, [None] * n

    def hook(event, i, _policy, tensor):
        (inputs if event == "start" else outputs)[i] = tensor

    _set_policies(transformer, reference)
    state.clear_grads()
    transformer.segment_hook = hook
    try:
        loss = loss_fn()
    finally:
        transformer.segment_hook = None
    loss.backward()
    missing = [i for i, o in enumerate(outputs) if o is None or o.grad is None]
    if missing:
        raise RuntimeError(f"no gradient reached the output of segment(s) {missing}")
    value = float(loss.numpy())
    xs = [x.detach() for x in inputs]  # keep the data, not the graph
    outs = [o.numpy().astype(np.float32) for o in outputs]
    grads = [o.grad.numpy().astype(np.float32) for o in outputs]
    del loss, inputs, outputs
    state.clear_grads()
    return value, xs, outs, grads


def _rerun(transformer, i, x, policy):
    """Output of segment i alone under `policy`, from input x (forward only)."""
    layer = transformer.layers.modules[i]
    with no_grad():
        out = run_segment(layer, x, policy, transformer.fp16_release_inputs)
    return out.numpy().astype(np.float32)


def estimate_segment_errors(transformer, loss_fn: LossFns, params: Optional[Iterable] = None,
                            reference: str = "none", eval_mode: bool = True) -> SegmentErrors:
    """First-order loss error of every half-precision policy on every segment of
    `transformer` (an nn.Transformer); see the module docstring.

    loss_fn: callable returning the scalar loss of one batch through the model that
        contains `transformer`, or a list of them (one per batch; the coefficients
        are averaged over the batches).
    params: the parameters whose .grad backward writes (default: the transformer's;
        pass model.parameters() to include e.g. embeddings); restored afterwards.
    reference: the float32 policy of the reference pass ("none" or "fp32").
    eval_mode: run with dropout off (needed whenever dropout > 0: the reruns must
        compute the same function as the reference).
    """
    if reference in HALF_PRECISION or reference not in SEGMENT_POLICIES:
        raise ValueError(f"the reference must be a float32 policy, got {reference!r}")
    fns = _as_list(loss_fn)
    n = transformer.num_layers
    precisions = sorted(set(HALF_PRECISION.values()))
    params = list(transformer.parameters() if params is None else params)
    losses, signed = [], []
    pert = [{prec: [] for prec in precisions} for _ in range(n)]
    with _Preserved(transformer, params, eval_mode) as state:
        for fn in fns:
            value, xs, outs, grads = _reference_pass(transformer, fn, reference, state)
            losses.append(value)
            batch = {}
            for prec in precisions:
                s = []
                for i in range(n):
                    d = _rerun(transformer, i, xs[i], prec).astype(np.float64) - outs[i]
                    s.append(float(np.sum(grads[i].astype(np.float64) * d)))
                    norm = float(np.linalg.norm(outs[i]))
                    pert[i][prec].append(float(np.linalg.norm(d)) / norm if norm > 0 else 0.0)
                batch[prec] = s
            signed.append(batch)
    policies = list(SEGMENT_POLICIES)
    error = []
    for i in range(n):
        row = {}
        for p in policies:
            prec = HALF_PRECISION.get(p)
            row[p] = 0.0 if prec is None else float(np.mean([abs(b[prec][i]) for b in signed]))
        error.append(row)
    return SegmentErrors(
        policies=policies, num_segments=n, device=params[0].device.name if params else "cpu",
        reference=reference, half_softmax=getattr(transformer, "half_softmax", None),
        loss=losses, signed=signed, error=error,
        perturbation=[{prec: float(np.mean(v)) for prec, v in row.items()} for row in pert],
    )


def measure_precision_error(transformer, loss_fn: LossFns, policies, params: Optional[Iterable] = None,
                            reference: str = "none", gradients: bool = True,
                            loss_scale: float = 1024.0, eval_mode: bool = True) -> dict:
    """The actual error of `policies` against the float32 `reference` on the same
    batches: {"loss_error": mean |L - L_ref|, "signed": [L - L_ref per batch],
    "loss": [L_ref per batch], "grad_error": mean relative L2 error of the
    gradients of `params` (None if gradients=False)}.

    Gradients are computed from loss * loss_scale and divided by it (a power of two,
    so exactly), as in training with a GradScaler: without scaling, small float16
    gradients underflow to zero.
    """
    fns = _as_list(loss_fn)
    params = list(transformer.parameters() if params is None else params)
    signed, ref_losses, grad_errors = [], [], []

    def run(fn, config, state):
        _set_policies(transformer, config)
        state.clear_grads()
        if not gradients:
            with no_grad():
                return float(fn().numpy()), None
        loss = fn()
        value = float(loss.numpy())
        (loss * loss_scale).backward()
        grads = [p.grad.numpy().astype(np.float64) / loss_scale for p in params]
        state.clear_grads()
        return value, grads

    with _Preserved(transformer, params, eval_mode) as state:
        for fn in fns:
            ref, g_ref = run(fn, reference, state)
            val, g = run(fn, policies, state)
            ref_losses.append(ref)
            signed.append(val - ref)
            if gradients:
                num = sum(float(np.sum((a - b) ** 2)) for a, b in zip(g, g_ref))
                den = sum(float(np.sum(b ** 2)) for b in g_ref)
                grad_errors.append(float(np.sqrt(num / den)) if den > 0 else 0.0)
    return {
        "loss_error": float(np.mean(np.abs(signed))),
        "signed": signed,
        "loss": ref_losses,
        "grad_error": float(np.mean(grad_errors)) if gradients else None,
    }
