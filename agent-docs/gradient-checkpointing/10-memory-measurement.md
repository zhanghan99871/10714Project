# Decision 10 — How to measure memory

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md) §6. Needed by the measurement tests (decision [07](07-cast-of-segment-inputs.md)
  follow-up) and the benchmark.

## Decision

A **Python counter in `NDArray.make`** ([python/needle/backend_ndarray/ndarray.py](../../python/needle/backend_ndarray/ndarray.py)):
each new buffer adds its size to a per-device counter and registers a `weakref.finalize` on its handle that subtracts
the size; the counter also keeps the peak. **One counter per device.** Three functions with PyTorch's names:
`memory_allocated()`, `max_memory_allocated()`, `reset_peak_memory_stats()`.

**Update 2026-11-14 (user):** the counter stays **always on**, with `weakref.finalize`. The benchmark measures its
effect on the measurements (decision [11](11-benchmark.md)); an on/off switch is added only if the effect is large.

**Update 2026-11-14 (measured, log [02](log/02-amp-benchmark.md)):** in A-B-B-A order the counter costs nothing measurable on the transformer (on/off 1.001 in float32, 1.002 in float16) and about 10% on the LSTM (1.097). **No on/off switch** (Claude, allowed by the user): the benchmark already times every run without the counter.

**Implementation** (log [01](log/01-memory-counter.md)): done, committed in `d011880`. The functions are in
`needle.backend_ndarray` (not exported at the top level of `ndl`); the device argument is optional (default
`default_device()`); one counter per device name. These details were chosen by Claude and not discussed.

## Facts

- The backends count nothing: CUDA arrays call `cudaMalloc` / `cudaFree` in the constructor / destructor
  ([ndarray_backend_cuda.cu:72-77](../../src/ndarray_backend_cuda.cu#L72-L77)); CPU arrays call `posix_memalign` / `free`
  ([ndarray_backend_cpu.cc:141-147](../../src/ndarray_backend_cpu.cc#L141-L147)). These are the only allocation calls in `src/`.
- Every array allocation goes through `NDArray.make` ([ndarray.py:225-231](../../python/needle/backend_ndarray/ndarray.py#L225-L231));
  no other Python file calls `device.Array(...)`.
- On numpy, cpu and cuda, the handle supports `weakref.finalize`, the finalizer runs when the last reference goes away,
  and the handle has `size` and `dtype` (checked on GPU 1).

## Options considered

| Option | Devices | Cost / problem |
|---|---|---|
| (a) C++ counter in the array constructor / destructor, exported by pybind | cpu, cuda (numpy separately) | C++ changes in two backends, rebuild |
| (b) CUDA free-memory query / `nvidia-smi` | cuda | includes other processes and the CUDA context; no peak. GPUs 2–7 run other jobs at 100%, GPU 0 has another process |
| **(c) Python counter in `NDArray.make` (chosen)** | numpy, cpu, cuda | ~15 lines, no rebuild |

## Comparison with PyTorch

PyTorch counts in its CUDA caching allocator and exposes `memory_allocated()` ("Return the current GPU memory occupied
by tensors in bytes"), `max_memory_allocated()` and `reset_peak_memory_stats()` (`torch/cuda/memory.py` L583, L600,
L416). needle has no caching allocator, so counting the buffers of `NDArray`s gives the bytes that needle holds.

## Justification

1. Same bytes as (a): every allocation goes through `NDArray.make`.
2. All three devices, no C++ change, no rebuild.
3. Familiar names (PyTorch).

## Limits

- Counts only `NDArray` buffers: not the CUDA context, not host copies made by `to_numpy`, not numpy temporaries inside
  the numpy backend's ops.
- One finalizer per new buffer. Measured cost (probes
  [agent-workspace/probe_counter_overhead.py](../../agent-workspace/probe_counter_overhead.py),
  [agent-workspace/probe_counter_overhead_2.py](../../agent-workspace/probe_counter_overhead_2.py)):
  ```
  NDArray.make 64x64: 2.59 us without counter, 5.42 us with counter (+2.82 us)
  x * 2.0 (64x64):    15.56 us without counter, 19.65 us with counter (+4.09 us)
  cuda 1024x1024 matmul + copy to host: 6988 us without counter, 7117 us with counter
  NDArray.make 64x64: none 2.42 us, finalize 5.02 us, weakref.ref 4.03 us
  ```
  About +26% on a tiny CPU op, about +2% on a large CUDA op. A `weakref.ref` with a callback (kept in a set) would cost
  about 1 µs less per allocation than `weakref.finalize`. Full test suite: 828 passed in 30.61 s before, 844 passed in
  34.12 s after (16 new tests; GPU 0 is shared with another job).
- Where the counter time goes (probes [agent-workspace/probe_counter_breakdown.py](../../agent-workspace/probe_counter_breakdown.py),
  [agent-workspace/probe_counter_breakdown_2.py](../../agent-workspace/probe_counter_breakdown_2.py), cpu device, 64×64 float32):
  ```
  NDArray.make((64, 64)) without counter       2.31 us
    of which: dev.Array(4096) (C++ malloc)     1.06 us
  counter parts:
    _memory_stats(device)                      0.17 us
    array.size  (prod(shape) via reduce)       0.21 us
    array.itemsize (np.dtype(str).itemsize)    0.25 us
    stats[0] += n; stats[1] = max(...)         0.14 us
  allocate + free a handle, no tracking:      1.17 us
    + weakref.finalize (register + callback): 2.27 us  (+1.10)
    + weakref.ref in a set (register + callback): 1.58 us  (+0.41)
  ```
  The counter is about ten Python-level steps per buffer; `weakref.finalize` alone is a pure-Python class
  (`weakref.py`) that builds an info object, a weak reference and a registry entry, and runs Python code again on
  free. `NDArray.make` itself is cheap because a 16 KB `posix_memalign` reuses freed blocks. The counter cost is fixed
  per buffer, independent of the buffer size.
