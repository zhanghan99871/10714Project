# Decision 10: How to measure memory

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/10-memory-measurement.md](../../agent-docs/gradient-checkpointing/10-memory-measurement.md).

## Decision

A counter in Python, in `NDArray.make`, counts memory. Each new buffer adds its size to the counter of its device. A finalizer subtracts the size when Python frees the buffer. The counter also keeps the peak value. There are three functions, with the same names as in PyTorch: `memory_allocated()`, `max_memory_allocated()` and `reset_peak_memory_stats()`.

**Update (2026-11-14):** the counter stays on at all times. The benchmark measures its effect (decision 11). We add an on/off switch only if the effect is large.

**Update (measured, log 02):** the counter has no measurable cost on the transformer, and about 10% on the LSTM. We do not add a switch, because the benchmark also measures each run without the counter. (Claude)

**Implementation:** done (log 01). The functions are in `needle.backend_ndarray`. Claude chose this detail without a discussion.

## Why

- The backends do not count memory.
- Every array allocation goes through `NDArray.make`. Thus, the Python counter sees all the buffers.
- It works on numpy, cpu and cuda, and it needs no C++ change.
- The other options are worse:
  - A counter in C++ needs changes in two backends and a rebuild.
  - The free GPU memory from CUDA includes other processes, and it has no peak value.

## Comparison with PyTorch

PyTorch counts memory in its CUDA allocator, and it gives the same three functions. Needle has no such allocator. Thus, needle counts the buffers of its arrays.

## Limits

- The counter counts only array buffers. It does not count the CUDA context, the host copies for `.numpy()`, or temporary numpy arrays in the numpy backend.
- The counter costs about 3 µs for each new buffer. This is about 26% more time for a very small op on the CPU, and about 2% for a large op on CUDA.
- The counter takes more time than `NDArray.make` itself. The counter does about ten Python steps for each buffer. The largest step is `weakref.finalize`, which is Python code that runs when the buffer is made and again when it is freed (1.1 µs). A small allocation in C++ takes only about 1 µs, because the allocator uses freed blocks again.
