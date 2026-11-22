"""Precision error of the segment policies, per segment, relative to batch noise.

Two measures of what running segment i (one TransformerLayer) in float16 instead
of float32 does to a training step on batch b:

loss      the first-order loss change
              s_bi = <grad_{o_i} L_b, o_i^fp16 - o_i^fp32>
          (Taylor expansion of the loss around the float32 segment output o_i,
          both outputs computed from the same float32 input)
gradient  the change of the parameter gradients, measured:
              delta_bi = g_b(segment i in float16) - g_b(float32)

Errors from different segments are combined as a sum of squares, i.e. treated as
uncorrelated (rounding errors are, to a good approximation; for the gradient,
a high-dimensional vector, their cross terms average out):

    loss_sq[i] = mean_b s_bi^2               grad_sq[i] = mean_b |delta_bi|^2
    E_b[(sum_i s_bi)^2]  ~ sum_i loss_sq[i]  E_b|sum_i delta_bi|^2 ~ sum_i grad_sq[i]

for the segments in half precision, and both are measured against the noise that
stochastic gradient descent already has between batches:

    loss_noise = Var_b(L_b)                  grad_noise = E_b|g_b - mean_b g_b|^2

predict(policies, metric) = sqrt(sum_i sq[i][p_i] / noise) is the RMS precision
error as a fraction of the batch noise's standard deviation; for the optimizer,
"predict <= eps" is the linear constraint

    sum_i coefficients(metric)[i][p_i] <= eps^2        (coefficients = sq / noise)

The coefficients are 0 for "fp32" and "none" (float32 numerics, bit-identical
gradients) and equal for "fp16" and "fp16_none" (bit-identical outputs and
gradients).

The reference is one step per batch with every segment at `reference` ("none" by
default: the float32 gradients bit for bit, least memory); its backward leaves
grad_{o_i} L_b on every segment boundary and the float32 parameter gradients g_b.
Each segment is then rerun alone in float16 from its recorded float32 input
(forward only) for s_bi and, if gradients=True, one step per segment with only
that segment in float16 gives delta_bi. Cost per batch: 1 + n steps (1 step and n
forwards without gradients). The batch noise needs at least two batches and is
only as good as their number (e.g. 8 batches give its standard deviation to
about +-25%).

Everything runs with the transformer in eval mode (dropout off), so all passes
compute the same function, and policies, segment hook, training flags and the
parameters' .grad are restored. The result records the model's half_softmax.
measure_precision_error() measures the actual errors of any configuration.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..autograd import no_grad
from ..nn.nn_transformer import SEGMENT_POLICIES, run_segment
from .profiler import _set_policies

__all__ = ["SegmentErrors", "estimate_segment_errors", "measure_precision_error"]

# the precision each policy's segment computes in, i.e. the rerun that gives its
# error; policies without an entry compute in float32 (error 0)
HALF_PRECISION = {"fp16": "fp16", "fp16_none": "fp16"}
METRICS = ("loss", "grad")

LossFns = Union[Callable, Sequence[Callable]]


@dataclass
class SegmentErrors:
    """Precision errors of the segment policies, per segment (see the module docstring)."""

    policies: List[str]
    num_segments: int
    device: str
    reference: str
    half_softmax: Optional[bool]
    loss: List[float]  # [batch] reference loss L_b
    signed: List[Dict[str, List[float]]]  # [batch] precision -> [segment] s_bi
    loss_noise: Optional[float]  # Var_b(L_b) (None with one batch)
    grad_sq_batches: Optional[List[Dict[str, List[float]]]] = None  # [batch] precision -> [segment] |delta_bi|^2
    grad_noise: Optional[float] = None  # E_b|g_b - mean g|^2
    grad_norm_sq: Optional[float] = None  # E_b|g_b|^2
    perturbation: List[Dict[str, float]] = field(default_factory=list)  # [segment][precision] |d_i| / |o_i|

    # ------------------------------------------------------------------ models
    def _check(self, policies) -> List[str]:
        policies = [policies] * self.num_segments if isinstance(policies, str) else list(policies)
        if len(policies) != self.num_segments:
            raise ValueError(f"expected {self.num_segments} policies, got {len(policies)}")
        for p in policies:
            if p not in self.policies:
                raise ValueError(f"unknown policy {p!r} (estimated: {self.policies})")
        return policies

    def _per_batch(self, metric):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if metric == "loss":
            return [{k: [v * v for v in vals] for k, vals in b.items()} for b in self.signed]
        if self.grad_sq_batches is None:
            raise ValueError("gradient errors were not estimated (gradients=False)")
        return self.grad_sq_batches

    def mean_sq(self, metric: str = "grad") -> List[Dict[str, float]]:
        """[segment][policy] mean over batches of the squared error (s_bi^2 or
        |delta_bi|^2); 0 for the float32 policies."""
        batches = self._per_batch(metric)
        out = []
        for i in range(self.num_segments):
            row = {}
            for p in self.policies:
                prec = HALF_PRECISION.get(p)
                row[p] = 0.0 if prec is None else float(np.mean([b[prec][i] for b in batches]))
            out.append(row)
        return out

    def noise(self, metric: str = "grad") -> float:
        """The batch noise of `metric`: Var_b(L_b) or E_b|g_b - mean g|^2."""
        value = self.loss_noise if metric == "loss" else self.grad_noise
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if value is None:
            raise ValueError(f"no {metric} noise: it needs at least two batches"
                             + ("" if metric == "loss" or self.grad_sq_batches is not None
                                else " and gradients=True"))
        return value

    def coefficients(self, metric: str = "grad") -> List[Dict[str, float]]:
        """[segment][policy] mean squared error / batch noise: the constraint
        "predict(policies, metric) <= eps" is sum_i coef[i][p_i] <= eps^2."""
        noise = self.noise(metric)
        return [{p: v / noise for p, v in row.items()} for row in self.mean_sq(metric)]

    def predict(self, policies, metric: str = "grad", relative: bool = True) -> float:
        """Modelled RMS precision error of `policies`: sqrt(sum_i mean_sq[i][p_i]),
        divided by the batch noise's standard deviation if relative."""
        policies = self._check(policies)
        total = sum(row[p] for row, p in zip(self.mean_sq(metric), policies))
        if relative:
            total /= self.noise(metric)
        return float(np.sqrt(total))

    def predict_signed(self, policies) -> float:
        """First-order RMS loss change without the uncorrelated-errors assumption:
        sqrt(mean_b (sum_i s_bi)^2) over the half-precision segments (not linear
        in the policies; a diagnostic of the sum-of-squares model)."""
        policies = self._check(policies)
        totals = [sum(b[HALF_PRECISION[p]][i] for i, p in enumerate(policies) if p in HALF_PRECISION) ** 2
                  for b in self.signed]
        return float(np.sqrt(np.mean(totals)))

    # ------------------------------------------------------------------ persistence
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
        head = (f"segment errors on {self.device} (reference={self.reference}, half_softmax="
                f"{self.half_softmax}, {len(self.loss)} batches, mean loss {np.mean(self.loss):.5g}")
        if self.loss_noise is not None:
            head += f", loss noise std {np.sqrt(self.loss_noise):.3g}"
        if self.grad_noise is not None:
            head += (f", gradient noise std {np.sqrt(self.grad_noise):.3g} "
                     f"(|g| {np.sqrt(self.grad_norm_sq):.3g})")
        lines = [head + ")"]
        metrics = [m for m in METRICS if m == "loss" or self.grad_sq_batches is not None]
        sq = {m: self.mean_sq(m) for m in metrics}
        for i in range(self.num_segments):
            cols = []
            for m in metrics:
                rms = np.sqrt(sq[m][i]["fp16"])
                col = f"{m} RMS {rms:.3e}"
                noise = self.loss_noise if m == "loss" else self.grad_noise
                if noise:
                    col += f" ({rms / np.sqrt(noise):.2e} of noise)"
                cols.append(col)
            pert = ", ".join(f"|d|/|o| {prec} {v:.2e}" for prec, v in self.perturbation[i].items())
            lines.append(f"  segment {i} (fp16 / fp16_none): " + " | ".join(cols) + (f" | {pert}" if pert else ""))
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


def _step(transformer, loss_fn, policies, params, state, loss_scale, hook=None):
    """One step under `policies`; returns (loss, parameter gradients as float64
    numpy, the loss tensor, the loss scale used). Gradients come from
    loss * loss_scale, divided by it (a power of two: exact); the scale is halved
    and the step redone while a float16 gradient overflows (callers keep using the
    reduced scale, as a GradScaler does)."""
    _set_policies(transformer, policies)
    scale = loss_scale
    for _ in range(256):
        state.clear_grads()
        transformer.segment_hook = hook
        try:
            loss = loss_fn()
        finally:
            transformer.segment_hook = None
        (loss * scale).backward()
        grads = [p.grad.numpy().astype(np.float64) / scale for p in params]
        if all(np.all(np.isfinite(g)) for g in grads):
            return float(loss.numpy()), grads, loss, scale
        scale /= 2
    raise RuntimeError("the gradients are not finite at any loss scale")


def _reference_pass(transformer, loss_fn, reference, params, state, loss_scale):
    """One step at `reference`; returns (loss, parameter gradients, [segment inputs,
    detached], [segment outputs as float32 numpy], [grad_{o_i} L as float64 numpy])."""
    n = transformer.num_layers
    inputs, outputs = [None] * n, [None] * n

    def hook(event, i, _policy, tensor):
        (inputs if event == "start" else outputs)[i] = tensor

    value, grads, loss, scale = _step(transformer, loss_fn, reference, params, state, loss_scale, hook)
    state.loss_scale = scale
    missing = [i for i, o in enumerate(outputs) if o is None or o.grad is None]
    if missing:
        raise RuntimeError(f"no gradient reached the output of segment(s) {missing}")
    xs = [x.detach() for x in inputs]  # keep the data, not the graph
    outs = [o.numpy().astype(np.float32) for o in outputs]
    out_grads = [o.grad.numpy().astype(np.float64) / scale for o in outputs]
    del loss, inputs, outputs
    state.clear_grads()
    return value, grads, xs, outs, out_grads


def _rerun(transformer, i, x, policy):
    """Output of segment i alone under `policy`, from input x (forward only)."""
    layer = transformer.layers.modules[i]
    with no_grad():
        out = run_segment(layer, x, policy, transformer.fp16_release_inputs)
    return out.numpy().astype(np.float32)


def _sq_dist(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    return float(sum(np.sum((x - y) ** 2) for x, y in zip(a, b)))


def estimate_segment_errors(transformer, loss_fn: LossFns, params: Optional[Iterable] = None,
                            reference: str = "none", gradients: bool = True,
                            loss_scale: float = 1024.0, eval_mode: bool = True,
                            grad_batches: Optional[int] = None) -> SegmentErrors:
    """Precision errors of every half-precision policy on every segment of
    `transformer` (an nn.Transformer), and the batch noise; see the module docstring.

    loss_fn: callable returning the scalar loss of one batch through the model that
        contains `transformer`, or a list of them, one per batch (at least two for
        the batch noise; the more, the better its estimate).
    params: the parameters whose gradients are compared and whose .grad backward
        writes (default: the transformer's; pass model.parameters() to include e.g.
        embeddings); their .grad is restored afterwards.
    reference: the float32 policy of the reference steps ("none" or "fp32").
    gradients: also measure the gradient errors (n extra steps per batch).
    grad_batches: measure the gradient errors on the first `grad_batches` batches
        only (default: all); the batch noise and the loss errors still use every
        batch. The gradient errors vary little between batches, so e.g. 2 of 8
        batches cost 8 + 2n steps instead of 8 + 8n.
    loss_scale: initial loss scale of the steps with float16 segments (halved while
        their gradients overflow), as a GradScaler would apply.
    eval_mode: dropout off (needed whenever dropout > 0: every pass must compute
        the same function).
    """
    if reference in HALF_PRECISION or reference not in SEGMENT_POLICIES:
        raise ValueError(f"the reference must be a float32 policy, got {reference!r}")
    fns = _as_list(loss_fn)
    if grad_batches is not None and grad_batches < 1:
        raise ValueError(f"grad_batches must be at least 1, got {grad_batches}")
    n = transformer.num_layers
    precisions = sorted(set(HALF_PRECISION.values()))
    params = list(transformer.parameters() if params is None else params)
    losses, signed, grad_sq = [], [], []
    pert = [{prec: [] for prec in precisions} for _ in range(n)]
    g_sum, g_sq = None, 0.0
    with _Preserved(transformer, params, eval_mode) as state:
        state.loss_scale = loss_scale
        for b, fn in enumerate(fns):
            measure_grads = gradients and (grad_batches is None or b < grad_batches)
            value, g_ref, xs, outs, out_grads = _reference_pass(transformer, fn, reference, params,
                                                                state, state.loss_scale)
            losses.append(value)
            g_sum = [g.copy() for g in g_ref] if g_sum is None else [a + g for a, g in zip(g_sum, g_ref)]
            g_sq += float(sum(np.sum(g * g) for g in g_ref))
            batch, batch_grad = {}, {}
            for prec in precisions:
                s, gs = [], []
                for i in range(n):
                    d = _rerun(transformer, i, xs[i], prec).astype(np.float64) - outs[i]
                    s.append(float(np.sum(out_grads[i] * d)))
                    norm = float(np.linalg.norm(outs[i]))
                    pert[i][prec].append(float(np.linalg.norm(d)) / norm if norm > 0 else 0.0)
                    if measure_grads:
                        config = [reference] * n
                        config[i] = prec
                        _, g, _, state.loss_scale = _step(transformer, fn, config, params, state,
                                                          state.loss_scale)
                        gs.append(_sq_dist(g, g_ref))
                batch[prec] = s
                batch_grad[prec] = gs
            signed.append(batch)
            if measure_grads:
                grad_sq.append(batch_grad)
            del xs, outs, out_grads
    B = len(fns)
    loss_noise = float(np.var(losses, ddof=1)) if B > 1 else None
    grad_noise = None
    if B > 1:
        mean_sq_norm = float(sum(np.sum(a * a) for a in g_sum)) / B
        grad_noise = max(0.0, (g_sq - mean_sq_norm) / (B - 1))
    return SegmentErrors(
        policies=list(SEGMENT_POLICIES), num_segments=n,
        device=params[0].device.name if params else "cpu", reference=reference,
        half_softmax=getattr(transformer, "half_softmax", None),
        loss=losses, signed=signed, loss_noise=loss_noise,
        grad_sq_batches=grad_sq if gradients else None,
        grad_noise=grad_noise if gradients else None,
        grad_norm_sq=g_sq / B if gradients else None,
        perturbation=[{prec: float(np.mean(v)) for prec, v in row.items()} for row in pert],
    )


def measure_precision_error(transformer, loss_fn: LossFns, policies, params: Optional[Iterable] = None,
                            reference: str = "none", gradients: bool = True,
                            loss_scale: float = 1024.0, eval_mode: bool = True) -> dict:
    """The actual errors of `policies` against the float32 `reference` on the same
    batches:

        signed      [batch] L_b - L_b^ref
        loss_rms    sqrt(mean_b (L_b - L_b^ref)^2)
        grad_sq     [batch] |g_b - g_b^ref|^2           (None if gradients=False)
        grad_rms    sqrt(mean_b |g_b - g_b^ref|^2)
        grad_rel    mean_b |g_b - g_b^ref| / |g_b^ref|
        loss        [batch] L_b^ref

    Gradients go through loss * loss_scale as in estimate_segment_errors."""
    fns = _as_list(loss_fn)
    params = list(transformer.parameters() if params is None else params)
    signed, ref_losses, grad_sq, grad_rel = [], [], [], []
    with _Preserved(transformer, params, eval_mode) as state:
        scale = loss_scale
        for fn in fns:
            if gradients:
                val, g, _, scale = _step(transformer, fn, policies, params, state, scale)
                ref, g_ref, _, scale = _step(transformer, fn, reference, params, state, scale)
                state.clear_grads()
                d2 = _sq_dist(g, g_ref)
                den = float(sum(np.sum(x * x) for x in g_ref))
                grad_sq.append(d2)
                grad_rel.append(float(np.sqrt(d2 / den)) if den > 0 else 0.0)
            else:
                with no_grad():
                    _set_policies(transformer, reference)
                    ref = float(fn().numpy())
                    _set_policies(transformer, policies)
                    val = float(fn().numpy())
            ref_losses.append(ref)
            signed.append(val - ref)
    return {
        "signed": signed,
        "loss_rms": float(np.sqrt(np.mean(np.square(signed)))),
        "grad_sq": grad_sq if gradients else None,
        "grad_rms": float(np.sqrt(np.mean(grad_sq))) if gradients else None,
        "grad_rel": float(np.mean(grad_rel)) if gradients else None,
        "loss": ref_losses,
    }
