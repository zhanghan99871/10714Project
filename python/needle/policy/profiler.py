"""Time and memory profile of the segments of a Transformer, per segment policy.

The profile is the cost table of the checkpoint-policy optimizer: for every segment
(one TransformerLayer, see nn.Transformer.set_policies) and every policy in
SEGMENT_POLICIES ("fp32", "fp16", "none", "fp16_none") it measures

time      the change of the step time (forward + backward) when only that segment
          switches from "fp32" to the policy (difference method: the baseline step
          has every segment at "fp32"; the cost of one segment is measured in the
          real, complete step, so kernel launch and synchronisation overheads are
          included). Every switched step is paired with an adjacent baseline step
          and time_delta is the median of the paired differences, so slow drift
          (clock boost, thermal throttling, other load) cancels; this matters
          because an error e in the baseline would move every prediction by
          (n - 1) * e. Rounds over all configurations are repeated `repeats` times
          after `warmup` rounds; CUDA work is synchronised around every timed step.
memory    from the needle memory counter (NDArray buffers), per phase of a step.
          One instrumented step per policy (every segment at that policy) records,
          for every segment i, relative to the memory when the phase starts:
          held[i][p]            bytes the segment's forward leaves allocated
          forward_peak[i][p]    peak while its forward runs
          backward_held[i][p]   bytes its backward adds (gradients kept until
                                backward returns)
          backward_peak[i][p]   peak while its backward runs (recomputation of
                                "none" / "fp16_none" segments happens here)
          and in `fixed` the same for the parts of the model before the first
          segment ("pre": embeddings) and after the last ("post": output layer,
          loss). These terms depend on the segment's own policy only, because
          segment boundaries are float32 whatever the policies.

The time model is linear in the per-segment choices; the peak is the maximum of
linear expressions, one per phase of the step:

    time(policies) = base_time + sum_i time_delta[i][p_i]
    peak(policies) = static + max_k (phase_k(policies)) + correction
    phase_k(policies) = const_k + sum_i coef_k[i][p_i]        (memory_phases())

with time_delta[i]["fp32"] = 0. The phases are the forward of the pre part, of each
segment and of the post part, then the backward of the post part, of each segment
(last to first) and of the pre part. Memory only grows through backward (needle
keeps every non-checkpointed node's gradient until backward returns), plus a
transient while each segment's backward runs; which phase is highest depends on
the whole configuration (e.g. an fp32 recompute in the middle can exceed a float16
recompute of the first segment, whose backward runs last), so a sum of per-segment
peak changes is not a valid model, but "peak <= budget" is still linear: one
constraint per phase. `correction` is the largest positive residual of the model
over every complete step measured while profiling (the timed single-switch steps
and the `calibration` configurations); peak_delta[i][p] (the change of the measured
peak when only segment i switches to p) is kept as a diagnostic.

Times depend on the device and its kernels: profile on the GPU used for training,
with tensor cores on (mma=True, the default), since "fp16" is only faster there.
The costs also depend on the model's options (e.g. Transformer.half_softmax, which
the profile records): re-profile when they change.
"""
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..autograd import backward_hook
from ..backend_ndarray import ndarray as nd
from ..nn.nn_transformer import SEGMENT_POLICIES

RECOMPUTE_POLICIES = ("none", "fp16_none")
# memory of the parts outside the segments (SegmentProfile.fixed): "pre" runs before
# the first segment (embeddings), "post" after the last (output layer, loss)
FIXED_TERMS = ("pre_forward_peak", "pre_held", "post_forward_peak", "post_held",
               "post_backward_peak", "post_backward_held", "pre_backward_peak")

__all__ = ["SegmentProfile", "profile_segments", "measure_step", "ProfileCache"]


@dataclass
class SegmentProfile:
    """Cost table of the segments of one model on one device for one input shape."""

    policies: List[str]
    num_segments: int
    device: str
    key: Optional[str]
    mma: Optional[bool]
    base_time: float  # seconds per step, every segment "fp32"
    time_delta: List[Dict[str, float]]  # [segment][policy] seconds vs "fp32"
    time_spread: List[Dict[str, float]]  # [segment][policy] half the min-max range
    forward_time: List[Dict[str, float]]  # [segment][policy] seconds (diagnostic)
    held: List[Dict[str, int]]  # [segment][policy] bytes kept by the forward
    forward_peak: List[Dict[str, int]]  # [segment][policy] bytes above the forward's start
    recompute: List[Dict[str, int]]  # [segment][recompute policy] bytes (diagnostic)
    static: int  # bytes
    base_peak: int  # bytes above static, every segment "fp32"
    peak_delta: List[Dict[str, int]]  # [segment][policy] bytes vs "fp32" (diagnostic)
    measured_peak: Dict[str, int] = field(default_factory=dict)  # config -> bytes above static
    correction: int = 0  # bytes
    residuals: Dict[str, int] = field(default_factory=dict)  # config -> measured - modelled
    half_softmax: Optional[bool] = None  # the transformer's half_softmax when profiled
    backward_held: Optional[List[Dict[str, int]]] = None  # [segment][policy] bytes added by its backward
    backward_peak: Optional[List[Dict[str, int]]] = None  # [segment][policy] bytes above the backward's start
    fixed: Optional[Dict[str, int]] = None  # pre / post parts, see FIXED_TERMS

    # ------------------------------------------------------------------ models
    def _check(self, policies: Sequence[str]) -> List[str]:
        policies = [policies] * self.num_segments if isinstance(policies, str) else list(policies)
        if len(policies) != self.num_segments:
            raise ValueError(f"expected {self.num_segments} policies, got {len(policies)}")
        for p in policies:
            if p not in self.policies:
                raise ValueError(f"policy {p!r} was not profiled (profiled: {self.policies})")
        return policies

    def predict_time(self, policies: Sequence[str]) -> float:
        """Modelled seconds per step (forward + backward) for `policies`."""
        policies = self._check(policies)
        return self.base_time + sum(self.time_delta[i][p] for i, p in enumerate(policies))

    def memory_phases(self) -> List[Tuple[str, int, List[Dict[str, int]]]]:
        """The phases of a step as (name, const, coef): the memory above `static` in
        that phase is const + sum_i coef[i][p_i], and the peak is the largest phase
        (plus `correction`). For the optimizer, "peak <= budget" is one linear
        constraint per phase."""
        if self.fixed is None:
            raise ValueError("this profile predates the phase model; profile again")
        n, P, f = self.num_segments, self.policies, self.fixed
        zero = {p: 0 for p in P}
        held = self.held
        phases = [("forward pre", f["pre_forward_peak"], [dict(zero) for _ in range(n)])]
        for j in range(n):
            coef = [dict(held[i]) if i < j else dict(self.forward_peak[i]) if i == j else dict(zero)
                    for i in range(n)]
            phases.append((f"forward segment {j}", f["pre_held"], coef))
        phases.append(("forward post", f["pre_held"] + f["post_forward_peak"], [dict(held[i]) for i in range(n)]))
        start = f["pre_held"] + f["post_held"]  # when backward starts
        phases.append(("backward post", start + f["post_backward_peak"], [dict(held[i]) for i in range(n)]))
        for j in reversed(range(n)):  # backward runs from the last segment to the first
            coef = [{p: held[i][p] + (self.backward_held[i][p] if i > j else
                                      self.backward_peak[i][p] if i == j else 0) for p in P}
                    for i in range(n)]
            phases.append((f"backward segment {j}", start + f["post_backward_held"], coef))
        phases.append(("backward pre", start + f["post_backward_held"] + f["pre_backward_peak"],
                       [{p: held[i][p] + self.backward_held[i][p] for p in P} for i in range(n)]))
        return phases

    def peak_phases(self, policies: Sequence[str]) -> Dict[str, int]:
        """Modelled bytes above `static` in every phase of a step (no correction)."""
        policies = self._check(policies)
        return {name: const + sum(coef[i][p] for i, p in enumerate(policies))
                for name, const, coef in self.memory_phases()}

    def _model_peak(self, policies: Sequence[str]) -> int:
        if self.fixed is None:
            # a profile saved before the phase model: sum of single-switch changes
            policies = self._check(policies)
            return self.base_peak + sum(self.peak_delta[i][p] for i, p in enumerate(policies))
        return max(self.peak_phases(policies).values())

    def predict_peak(self, policies: Sequence[str], absolute: bool = True) -> int:
        """Modelled peak bytes of a step; absolute=False gives the bytes above `static`."""
        peak = self._model_peak(policies) + self.correction
        return peak + self.static if absolute else peak

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentProfile":
        d = dict(d)
        if d["recompute"] and not isinstance(d["recompute"][0], dict):
            # profiles saved before "fp16_none": one number per segment, for "none"
            d["recompute"] = [{"none": r} if "none" in d["policies"] else {} for r in d["recompute"]]
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "SegmentProfile":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        mib = lambda b: f"{b / 2**20:.1f}"
        lines = [f"segment profile on {self.device} (mma={self.mma}, half_softmax={self.half_softmax}, "
                 f"key={self.key}): "
                 f"base step {self.base_time * 1e3:.2f} ms, static {mib(self.static)} MiB, "
                 f"base peak {mib(self.base_peak)} MiB, correction {self.correction / 2**20:+.2f} MiB"]
        if self.fixed is not None:
            f = self.fixed
            lines.append(f"  pre (embedding): forward peak {mib(f['pre_forward_peak'])}, held {mib(f['pre_held'])}, "
                         f"backward peak {mib(f['pre_backward_peak'])} MiB | post (output, loss): forward peak "
                         f"{mib(f['post_forward_peak'])}, held {mib(f['post_held'])}, backward peak "
                         f"{mib(f['post_backward_peak'])}, held {mib(f['post_backward_held'])} MiB")
        width = max(len(p) for p in self.policies)
        for i in range(self.num_segments):
            lines.append(f"  segment {i}:")
            for p in self.policies:
                line = (f"    {p:<{width}}  time {self.time_delta[i][p] * 1e3:+8.2f} ms | forward peak "
                        f"{mib(self.forward_peak[i][p]):>8}, held {mib(self.held[i][p]):>8} MiB")
                if self.backward_peak is not None:
                    line += (f" | backward peak {mib(self.backward_peak[i][p]):>8}, held "
                             f"{mib(self.backward_held[i][p]):>8} MiB")
                lines.append(line + f" | step peak {self.peak_delta[i][p] / 2**20:+9.1f} MiB")
        return "\n".join(lines)


def _config_name(policies: Sequence[str]) -> str:
    return ",".join(policies)


class _Profiling:
    """Saves and restores everything profiling changes on the model and device."""

    def __init__(self, transformer, params, device, mma):
        self.transformer = transformer
        self.params = list(params)
        self.device = device
        self.mma = mma

    def __enter__(self):
        tr = self.transformer
        self.saved_policies = None if tr.policies is None else list(tr.policies)
        self.saved_hook = tr.segment_hook
        tr.segment_hook = None  # a user's hook must not run inside the timed steps
        self.saved_grads = [getattr(p, "grad", None) for p in self.params]
        self.saved_mma = None
        mma_enabled = getattr(self.device.mod, "mma_enabled", None) if self.device.mod else None
        if mma_enabled is not None and self.mma is not None:
            self.saved_mma = mma_enabled()
            self.device.mod.set_mma_enabled(bool(self.mma))
        return self

    def __exit__(self, *exc):
        tr = self.transformer
        tr.set_policies(self.saved_policies)
        tr.segment_hook = self.saved_hook
        for p, g in zip(self.params, self.saved_grads):
            p.grad = g
        if self.saved_mma is not None:
            self.device.mod.set_mma_enabled(self.saved_mma)
        return False

    def clear_grads(self):
        for p in self.params:
            p.grad = None


def _device_of(transformer, params) -> nd.BackendDevice:
    if getattr(transformer, "device", None) is not None:
        return transformer.device
    params = list(params)
    if params:
        return params[0].device
    return nd.default_device()


def measure_step(transformer, loss_fn: Callable, policies: Sequence[str],
                 params: Optional[Iterable] = None, warmup: int = 1, repeats: int = 3,
                 device=None, mma: Optional[bool] = True) -> dict:
    """Run `warmup` + `repeats` complete steps (loss_fn() + backward) with `policies`
    and return {"time": median seconds, "times": [...], "peak": peak bytes above
    the memory held before the step, "static": that memory}. Restores the model's
    policies, hook and the parameters' .grad afterwards."""
    params = list(transformer.parameters() if params is None else params)
    device = device or _device_of(transformer, params)
    with _Profiling(transformer, params, device, mma) as prof:
        times, static, peak = _time_config(transformer, loss_fn, policies, prof, device, warmup, repeats)
    return {"time": statistics.median(times), "times": times, "peak": peak, "static": static}


def _set_policies(transformer, policies):
    transformer.set_policies(policies if isinstance(policies, str) else list(policies))


def _step(transformer, loss_fn, prof, device):
    """One complete step (loss_fn() + backward) with the current policies; returns
    (seconds, bytes held before the step, peak bytes above that)."""
    prof.clear_grads()
    gc.collect()
    device.synchronize()
    before = nd.memory_allocated(device)
    nd.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = loss_fn()
    loss.backward()
    device.synchronize()
    elapsed = time.perf_counter() - start
    peak = nd.max_memory_allocated(device) - before
    del loss
    return elapsed, before, peak


def _time_config(transformer, loss_fn, policies, prof, device, warmup, repeats):
    """Timed complete steps; returns (times of the `repeats` steps, static bytes
    before a step, highest peak above static over all steps)."""
    _set_policies(transformer, policies)
    times, static, peak = [], None, 0
    for step in range(warmup + repeats):
        elapsed, before, pk = _step(transformer, loss_fn, prof, device)
        static = before if static is None else min(static, before)
        peak = max(peak, pk)
        if step >= warmup:
            times.append(elapsed)
    prof.clear_grads()
    return times, static, peak


def _time_switched(transformer, loss_fn, n, switched, prof, device, warmup, repeats):
    """Paired timing of the baseline (every segment "fp32") against configurations
    with one segment i switched to policy p, for every (i, p) in `switched`.

    Each round runs, for every (i, p), a baseline step and the switched step back to
    back (in alternating order from round to round); the first `warmup` rounds are
    discarded. Returns (baseline times, {(i, p): paired differences}, static bytes,
    baseline peak, {(i, p): peak})."""
    base = ["fp32"] * n
    base_times, diffs = [], {c: [] for c in switched}
    static, base_peak, peaks = None, 0, {c: 0 for c in switched}
    pairs = switched if switched else [None]
    for r in range(warmup + repeats):
        for c in pairs:
            order = ["base", "switched"] if r % 2 == 0 else ["switched", "base"]
            t = {}
            for which in order:
                if which == "base" or c is None:
                    _set_policies(transformer, base)
                else:
                    config = list(base)
                    config[c[0]] = c[1]
                    _set_policies(transformer, config)
                elapsed, before, pk = _step(transformer, loss_fn, prof, device)
                t[which] = elapsed
                static = before if static is None else min(static, before)
                if which == "base" or c is None:
                    base_peak = max(base_peak, pk)
                else:
                    peaks[c] = max(peaks[c], pk)
            if r >= warmup:
                base_times.append(t["base"])
                if c is not None:
                    diffs[c].append(t["switched"] - t["base"])
    prof.clear_grads()
    return base_times, diffs, static, base_peak, peaks


def _memory_step(transformer, loss_fn, policies, prof, device):
    """One complete step with the memory of every phase recorded (see the module
    docstring). Returns (per-segment terms {"held", "forward_peak", "forward_time",
    "backward_held", "backward_peak"} as lists, fixed terms, the step's peak)."""
    n = transformer.num_layers
    seg = {k: [0] * n for k in ("held", "forward_peak", "backward_held", "backward_peak")}
    seg["forward_time"] = [0.0] * n
    fixed = {}
    state = {"peak": 0}
    seg_input = {}  # segment index -> id of its input
    seg_output = {}  # id of a segment's output -> segment index

    def begin():
        device.synchronize()
        state["mem"] = nd.memory_allocated(device)
        state["t"] = time.perf_counter()
        nd.reset_peak_memory_stats(device)

    def end():
        """(bytes added, peak above the start) of the running phase."""
        device.synchronize()
        state["elapsed"] = time.perf_counter() - state["t"]
        top = nd.max_memory_allocated(device)
        state["peak"] = max(state["peak"], top - state["before"])
        return nd.memory_allocated(device) - state["mem"], top - state["mem"]

    def segment_hook(event, i, _policy, tensor):
        if event == "start":
            if i == 0:
                fixed["pre_held"], fixed["pre_forward_peak"] = end()
            seg_input[i] = id(tensor)
            begin()
        else:
            seg["held"][i], seg["forward_peak"][i] = end()
            seg["forward_time"][i] = state["elapsed"]
            seg_output[id(tensor)] = i
            begin()

    order = []  # the backward phases in the order they were entered

    def node_hook(node):
        j = seg_output.get(id(node))
        if j is None and n and id(node) == seg_input[0]:
            j = -1  # the input of the first segment: the pre part's backward starts
        if j is None:
            return
        added, top = end()
        prev = order[-1] if order else n  # n: the post part
        if prev == n:
            fixed["post_backward_held"], fixed["post_backward_peak"] = added, top
        else:
            seg["backward_held"][prev], seg["backward_peak"][prev] = added, top
        order.append(j)
        begin()

    _set_policies(transformer, policies)
    prof.clear_grads()
    gc.collect()
    device.synchronize()
    state["before"] = nd.memory_allocated(device)
    begin()
    transformer.segment_hook = segment_hook
    try:
        loss = loss_fn()
        fixed["post_held"], fixed["post_forward_peak"] = end()
        begin()
        with backward_hook(node_hook):
            loss.backward()
        _, fixed["pre_backward_peak"] = end()
    finally:
        transformer.segment_hook = None
    if order != list(reversed(range(-1, n))):
        raise RuntimeError(f"segment boundaries not found in backward in the expected order: {order}")
    del loss
    prof.clear_grads()
    gc.collect()
    return seg, fixed, state["peak"]


def profile_segments(transformer, loss_fn: Callable, params: Optional[Iterable] = None,
                     policies: Sequence[str] = SEGMENT_POLICIES, warmup: int = 1,
                     repeats: int = 3, device=None, mma: Optional[bool] = True,
                     key: Optional[str] = None, calibration: Optional[List[Sequence[str]]] = None
                     ) -> SegmentProfile:
    """Profile every segment of `transformer` (an nn.Transformer) under every policy.

    loss_fn: callable returning the scalar loss of one batch through the model that
        contains `transformer` (it is called many times; the batch should be fixed).
    params: the parameters whose .grad backward writes (default: the transformer's;
        pass model.parameters() to include e.g. embeddings). Their .grad, and the
        transformer's policies and segment hook, are restored afterwards.
    policies: the policies to profile; must contain "fp32" (the baseline).
    warmup, repeats: rounds over the timed configurations; each round times every
        switched configuration next to a baseline step (medians over `repeats`).
    mma: force CUDA tensor cores on (True) / off (False) while profiling; None
        leaves the setting alone. Ignored on cpu / numpy. The result's `mma` says
        whether fp16 matmuls actually used tensor cores (False on GPUs before sm_70).
    key: label of what was profiled (e.g. the input shape), stored in the result.
    calibration: policy configurations whose measured peaks calibrate the memory
        model's correction term, in addition to the timed single-switch steps
        (default: every policy on every segment).
    """
    policies = list(policies)
    if "fp32" not in policies:
        raise ValueError("the profiled policies must include the 'fp32' baseline")
    for p in policies:
        if p not in SEGMENT_POLICIES:
            raise ValueError(f"unknown segment policy {p!r}; expected one of {SEGMENT_POLICIES}")
    params = list(transformer.parameters() if params is None else params)
    device = device or _device_of(transformer, params)
    n = transformer.num_layers

    with _Profiling(transformer, params, device, mma) as prof:
        # whether "fp16" matmuls really ran on tensor cores: switched on *and*
        # supported by the GPU (sm_70+); None on cpu / numpy
        mma_state = None
        if device.mod is not None and hasattr(device.mod, "mma_enabled"):
            mma_state = bool(device.mod.mma_enabled()) and bool(device.mod.mma_available("float16"))

        # memory per phase: one instrumented step with every segment at one policy
        terms = {k: [dict() for _ in range(n)]
                 for k in ("held", "forward_peak", "forward_time", "backward_held", "backward_peak")}
        fixed = {}
        for p in policies:
            seg, fx, _ = _memory_step(transformer, loss_fn, p, prof, device)
            for k, values in seg.items():
                for i in range(n):
                    terms[k][i][p] = values[i]
            for k, v in fx.items():  # the same in every run; keep the largest
                fixed[k] = max(fixed.get(k, v), v)
        recompute = [{p: terms["forward_peak"][i][p] for p in RECOMPUTE_POLICIES if p in policies}
                     for i in range(n)]

        # time and peak: one segment switched at a time, paired with baseline steps
        switched = [(i, p) for i in range(n) for p in policies if p != "fp32"]
        base_times, diffs, static, base_peak, peaks = _time_switched(
            transformer, loss_fn, n, switched, prof, device, warmup, repeats)
        base_time = statistics.median(base_times)
        delta = [{"fp32": 0.0} for _ in range(n)]
        spread = [{"fp32": (max(base_times) - min(base_times)) / 2} for _ in range(n)]
        pdelta = [{"fp32": 0} for _ in range(n)]
        # measured peaks of complete steps, to calibrate the correction term
        measured = {_config_name(["fp32"] * n): base_peak}
        for (i, p), d in diffs.items():
            delta[i][p] = statistics.median(d)
            spread[i][p] = (max(d) - min(d)) / 2
            pdelta[i][p] = peaks[(i, p)] - base_peak
            config = ["fp32"] * n
            config[i] = p
            measured[_config_name(config)] = peaks[(i, p)]

        configs = calibration if calibration is not None else [[p] * n for p in policies]
        for config in configs:
            _, st, pk = _time_config(transformer, loss_fn, config, prof, device, 0, 1)
            static = min(static, st)
            measured[_config_name(config)] = pk

    profile = SegmentProfile(
        policies=policies, num_segments=n, device=device.name, key=key, mma=mma_state,
        base_time=base_time, time_delta=delta, time_spread=spread,
        forward_time=terms["forward_time"], held=terms["held"], forward_peak=terms["forward_peak"],
        recompute=recompute, static=static, half_softmax=getattr(transformer, "half_softmax", None),
        base_peak=base_peak, peak_delta=pdelta, measured_peak=measured,
        backward_held=terms["backward_held"], backward_peak=terms["backward_peak"],
        fixed={k: fixed[k] for k in FIXED_TERMS},
    )
    residual = {name: pk - profile._model_peak(name.split(",")) for name, pk in measured.items()}
    profile.correction = max(0, max(residual.values())) if residual else 0
    profile.residuals = residual
    return profile


class ProfileCache:
    """Profiles keyed by what they depend on (e.g. batch size and sequence length).

    The weights do not change time or memory, so a profile stays valid for the
    whole training run; re-profile only when the input shape changes."""

    def __init__(self):
        self._profiles: Dict[str, SegmentProfile] = {}

    def __contains__(self, key) -> bool:
        return str(key) in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def get(self, key) -> Optional[SegmentProfile]:
        return self._profiles.get(str(key))

    def put(self, key, profile: SegmentProfile) -> None:
        self._profiles[str(key)] = profile

    def get_or_profile(self, key, profile_fn: Callable[[], SegmentProfile]) -> SegmentProfile:
        key = str(key)
        if key not in self._profiles:
            self._profiles[key] = profile_fn()
        return self._profiles[key]
