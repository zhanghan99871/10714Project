# Log 01 — Memory counter (order of work, step 1)

- Date: 2026-11-14. Branch: `gradient-checkpointing` (from `main` at `83ae951`). Status: **implemented, committed** in `d011880` (log [02](02-amp-benchmark.md), "Commit").
- Decision: [10-memory-measurement.md](../10-memory-measurement.md).

## What was done

- [python/needle/backend_ndarray/ndarray.py](../../../python/needle/backend_ndarray/ndarray.py) (+41 lines): `import weakref`;
  `_MEMORY_STATS` (device name → `[allocated, peak]`), `_memory_stats()`, `_track_allocation()`, `_track_free()`;
  public `memory_allocated()`, `max_memory_allocated()`, `reset_peak_memory_stats()`; `NDArray.make` calls
  `_track_allocation(array)` after it allocates a buffer.
- [tests/unit/test_memory_stats.py](../../../tests/unit/test_memory_stats.py): 16 tests on numpy / cpu / cuda — allocation
  and free, bfloat16 = 2 bytes, views share their buffer, peak and reset, tensor-op outputs are counted, devices are
  separate. An autouse fixture disables the cycle garbage collector during each test (a collection could free buffers
  of earlier tests).

## Results

- New tests: `16 passed in 0.33s`.
- Full suite: `844 passed in 34.12s` (baseline `828 passed in 30.61s`; GPU 0 is shared with another job).
  Log: [agent-workspace/pytest_after_memory_counter.log](../../../agent-workspace/pytest_after_memory_counter.log).

## Measurements

Counter cost ([probe_counter_overhead.py](../../../agent-workspace/probe_counter_overhead.py),
[probe_counter_overhead_2.py](../../../agent-workspace/probe_counter_overhead_2.py)):
```
NDArray.make 64x64: 2.59 us without counter, 5.42 us with counter (+2.82 us)
x * 2.0 (64x64):    15.56 us without counter, 19.65 us with counter (+4.09 us)
cuda 1024x1024 matmul + copy to host: 6988 us without counter, 7117 us with counter
NDArray.make 64x64: none 2.42 us, finalize 5.02 us, weakref.ref 4.03 us
```
Breakdown ([probe_counter_breakdown.py](../../../agent-workspace/probe_counter_breakdown.py),
[probe_counter_breakdown_2.py](../../../agent-workspace/probe_counter_breakdown_2.py)): `weakref.finalize` +1.10 µs per
buffer (register + callback, pure Python); `size` 0.21, `itemsize` 0.25, `_memory_stats` 0.17, arithmetic 0.14 µs.

Allocations per `TransformerLayer` forward + backward on cuda
([probe_alloc_count_amp.py](../../../agent-workspace/probe_alloc_count_amp.py)):
```
float32 : 351 allocations in forward + backward
float16 : 383 allocations in forward + backward
bfloat16: 383 allocations in forward + backward
```
The counter adds a fixed time per buffer to both float32 and AMP runs, so it lowers the measured AMP speedup a little;
it does not reverse the order unless AMP saves almost no time.

## Decisions during this step

| # | Decision | Who | Recorded in |
|---|---|---|---|
| 1 | Work in the branch `gradient-checkpointing`; `main` stays at `83ae951` | user | [00-plan.md](../00-plan.md) working rules |
| 2 | Implementation steps start only when the user says "start" | user | [00-plan.md](../00-plan.md) working rules |
| 3 | The counter stays always on (`weakref.finalize`); the benchmark measures its effect; an on/off switch only if the effect is large | user | [10](../10-memory-measurement.md) (updated), [11](../11-benchmark.md) (updated) |
| 4 | The functions are in `needle.backend_ndarray` (not exported at the top level of `ndl`); the device argument is optional (default `default_device()`); one counter per device name | Claude, not discussed | [10](../10-memory-measurement.md) (updated) |

## Corrections

- Decision 10 first said the counter cost is "small compared with an op". Measured: about +26% for a tiny CPU op,
  about +2% for a large CUDA op. Decision 10 is corrected.
- Step 1 was started before the user said "start". The user kept the work (decision 3 above).

## Next

Step 2, the AMP benchmark ([11](../11-benchmark.md)) — waits for "start".
