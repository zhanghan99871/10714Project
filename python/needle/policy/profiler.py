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
memory    from the needle memory counter (NDArray buffers), with the same
          difference method: peak_delta[i][p] is the change of the step's peak
          when only segment i switches to p. Also recorded:
          static               bytes held before the step (parameters, inputs,
                               optimizer state, ...)
          base_peak            peak above static of the all-"fp32" step
          held[i][p], forward_peak[i][p], recompute[i][p]
                               diagnostics: bytes segment i holds when it returns,
                               its extra peak while running forward, and (for the
                               recompute policies "none" / "fp16_none") the memory
                               it re-materialises in backward

The models are linear in the per-segment choices, as the optimizer needs:

    time(policies) = base_time + sum_i time_delta[i][p_i]
    peak(policies) = static + base_peak + sum_i peak_delta[i][p_i] + correction

with time_delta[i]["fp32"] = peak_delta[i]["fp32"] = 0. The peak is additive in
needle because backward keeps the .grad of every node outside checkpoint segments
until it ends: memory grows during backward and peaks at its end, when every
segment's kept activations and gradients are alive and every recomputed segment
has already been released again (a recompute happens earlier, with fewer gradients
alive). `correction` is the largest positive residual of the model on calibration
configs (complete steps with every segment at one policy, by default); it absorbs
peaks the additive model misses, e.g. if a recompute transient ever set the peak.

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
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ..backend_ndarray import ndarray as nd
from ..nn.nn_transformer import SEGMENT_POLICIES

RECOMPUTE_POLICIES = ("none", "fp16_none")

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
    held: List[Dict[str, int]]  # [segment][policy] bytes (diagnostic)
    forward_peak: List[Dict[str, int]]  # [segment][policy] bytes (diagnostic)
    recompute: List[Dict[str, int]]  # [segment][recompute policy] bytes (diagnostic)
    static: int  # bytes
    base_peak: int  # bytes above static, every segment "fp32"
    peak_delta: List[Dict[str, int]]  # [segment][policy] bytes vs "fp32"
    measured_peak: Dict[str, int] = field(default_factory=dict)  # config -> bytes above static
    correction: int = 0  # bytes
    residuals: Dict[str, int] = field(default_factory=dict)  # config -> measured - modelled
    half_softmax: Optional[bool] = None  # the transformer's half_softmax when profiled

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

    def segment_memory(self, i: int, policy: str) -> int:
        """The memory term of segment i under `policy` (bytes vs "fp32")."""
        return self.peak_delta[i][policy]

    def _model_peak(self, policies: Sequence[str]) -> int:
        policies = self._check(policies)
        return self.base_peak + sum(self.segment_memory(i, p) for i, p in enumerate(policies))

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
        lines = [f"segment profile on {self.device} (mma={self.mma}, half_softmax={self.half_softmax}, "
                 f"key={self.key}): "
                 f"base step {self.base_time * 1e3:.2f} ms, static {self.static / 2**20:.1f} MiB, "
                 f"base peak {self.base_peak / 2**20:.1f} MiB, correction {self.correction / 2**20:+.2f} MiB"]
        for i in range(self.num_segments):
            cols = []
            for p in self.policies:
                cols.append(f"{p}: {self.time_delta[i][p] * 1e3:+.2f} ms, "
                            f"peak {self.peak_delta[i][p] / 2**20:+.2f} MiB")
            lines.append(f"  segment {i}: " + " | ".join(cols)
                         + "".join(f" | recompute {p} {b / 2**20:.2f} MiB"
                                   for p, b in self.recompute[i].items()))
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


def _segment_memory(transformer, loss_fn, policy, prof, device):
    """One forward with every segment at `policy`, recording per segment the bytes
    held when it returns, its extra peak while running, and its forward time."""
    n = transformer.num_layers
    held, peak, ftime = [0] * n, [0] * n, [0.0] * n
    state = {}

    def hook(event, i, _policy):
        device.synchronize()
        if event == "start":
            state["mem"] = nd.memory_allocated(device)
            nd.reset_peak_memory_stats(device)
            state["t"] = time.perf_counter()
        else:
            ftime[i] = time.perf_counter() - state["t"]
            held[i] = nd.memory_allocated(device) - state["mem"]
            peak[i] = nd.max_memory_allocated(device) - state["mem"]

    transformer.set_policies(policy)
    prof.clear_grads()
    gc.collect()
    transformer.segment_hook = hook
    try:
        loss = loss_fn()
    finally:
        transformer.segment_hook = None
    del loss
    gc.collect()
    return held, peak, ftime


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
        model's correction term (default: every policy on every segment).
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

        # memory per segment and policy (forward with every segment at one policy)
        held = [dict() for _ in range(n)]
        fpeak = [dict() for _ in range(n)]
        ftime = [dict() for _ in range(n)]
        for p in policies:
            h, pk, ft = _segment_memory(transformer, loss_fn, p, prof, device)
            for i in range(n):
                held[i][p], fpeak[i][p], ftime[i][p] = h[i], pk[i], ft[i]
        recompute = [{p: fpeak[i][p] for p in RECOMPUTE_POLICIES if p in policies} for i in range(n)]

        # time and peak: one segment switched at a time, paired with baseline steps
        switched = [(i, p) for i in range(n) for p in policies if p != "fp32"]
        base_times, diffs, static, base_peak, peaks = _time_switched(
            transformer, loss_fn, n, switched, prof, device, warmup, repeats)
        base_time = statistics.median(base_times)
        delta = [{"fp32": 0.0} for _ in range(n)]
        spread = [{"fp32": (max(base_times) - min(base_times)) / 2} for _ in range(n)]
        pdelta = [{"fp32": 0} for _ in range(n)]
        for (i, p), d in diffs.items():
            delta[i][p] = statistics.median(d)
            spread[i][p] = (max(d) - min(d)) / 2
            pdelta[i][p] = peaks[(i, p)] - base_peak

        # measured peaks of complete steps, to calibrate the correction term
        configs = calibration if calibration is not None else [[p] * n for p in policies]
        measured = {}
        for config in configs:
            _, st, pk = _time_config(transformer, loss_fn, config, prof, device, 0, 1)
            static = min(static, st)
            measured[_config_name(config)] = pk

    profile = SegmentProfile(
        policies=policies, num_segments=n, device=device.name, key=key, mma=mma_state,
        base_time=base_time, time_delta=delta, time_spread=spread, forward_time=ftime,
        held=held, forward_peak=fpeak, recompute=recompute, static=static,
        half_softmax=getattr(transformer, "half_softmax", None),
        base_peak=base_peak, peak_delta=pdelta, measured_peak=measured,
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
