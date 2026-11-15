# Log 02 — AMP benchmark (order of work, step 2)

- Date: 2026-11-14. Branch: `gradient-checkpointing`. Status: **implemented, run, committed** (see "Commit").
- Decision: [11-benchmark.md](../11-benchmark.md). The user said "start" for steps 2–4 and allowed Claude to make the
  new decisions.

## What was done

- [apps/benchmark.py](../../../apps/benchmark.py): one configuration per process (`--model`, `--size`, `--amp`);
  `--all` runs the 12 configurations (2 models × 2 sizes × 3 dtypes) in subprocesses and prints a table. A run trains a
  `LanguageModel` with `epoch_general_ptb` on random tokens: 2 warm-up steps, 10 timed steps with the memory counter
  on (peak, time, loss), 10 timed steps with the counter replaced by a no-op (time only).
- Command (GPU 1, the free GPU):
  `CUDA_VISIBLE_DEVICES=1 python apps/benchmark.py --all --steps 10 --warmup 2 --out agent-workspace/benchmark/amp.json`
  Raw results: [agent-workspace/benchmark/amp.json](../../../agent-workspace/benchmark/amp.json),
  [agent-workspace/benchmark/amp.log](../../../agent-workspace/benchmark/amp.log).

Sizes (vocabulary 1000 everywhere):

| model | size | embedding | hidden | layers | batch | seq_len | parameters |
|---|---|---|---|---|---|---|---|
| transformer | activations | 256 | 1024 | 4 | 32 | 256 | 4.2 M |
| transformer | weights | 1024 | 4096 | 4 | 4 | 16 | 41.9 M |
| lstm | activations | 256 | 512 | 2 | 64 | 64 | 4.4 M |
| lstm | weights | 1024 | 2048 | 2 | 4 | 8 | 61.8 M |

## Results

| model | size | amp | peak MB | peak vs fp32 | ms/step (counter on) | ms/step (counter off) | speedup vs fp32 (counter off) | loss | skipped |
|---|---|---|---|---|---|---|---|---|---|
| transformer | activations | float32 | 10543 | 1.00 | 3929 | 3383 | 1.00 | 14.9774 | 0 |
| transformer | activations | float16 | 10618 | 1.01 | 2526 | 2503 | 1.35 | 14.9775 | 0 |
| transformer | activations | bfloat16 | 10618 | 1.01 | 2516 | 2512 | 1.35 | 14.9776 | 0 |
| transformer | weights | float32 | 702 | 1.00 | 256 | 246 | 1.00 | 29.5101 | 0 |
| transformer | weights | float16 | 874 | 1.24 | 280 | 273 | 0.90 | 26.8490 | 3 |
| transformer | weights | bfloat16 | 874 | 1.24 | 216 | 202 | 1.22 | 29.0498 | 0 |
| lstm | activations | float32 | 2662 | 1.00 | 2060 | 1893 | 1.00 | 6.9095 | 0 |
| lstm | activations | float16 | 2380 | 0.89 | 1844 | 1692 | 1.12 | 6.9095 | 0 |
| lstm | activations | bfloat16 | 2380 | 0.89 | 1721 | 1537 | 1.23 | 6.9095 | 0 |
| lstm | weights | float32 | 2607 | 1.00 | 467 | 452 | 1.00 | 6.9127 | 0 |
| lstm | weights | float16 | 2427 | 0.93 | 549 | 540 | 0.84 | 6.9127 | 0 |
| lstm | weights | bfloat16 | 2427 | 0.93 | 454 | 441 | 1.02 | 6.9127 | 0 |

Speedups use the counter-off column (see finding 6).

## Findings

1. **AMP saves no memory on the activation-heavy transformer (1.01×).** `nn.Linear` adds the float32 bias in a separate
   op, and the "promote" policy turns the float16 matmul result back into float32
   ([nn_basic.py:96-106](../../../python/needle/nn/nn_basic.py#L96-L106); probe
   [probe_linear_dtype.py](../../../agent-workspace/probe_linear_dtype.py): `matmul output: float16 | Linear output:
   float32`). After one `TransformerLayer` forward (benchmark shapes) the graph holds 1281 MB in float32 and 1242 MB
   under float16 autocast; 80 of 101 non-attention buffers and 5 of 10 attention-sized buffers stay float32 (probe
   [probe_amp_graph_memory.py](../../../agent-workspace/probe_amp_graph_memory.py)).
2. **AMP speeds up the activation-heavy transformer by about 1.35×** (counter-off column; the A-B-B-A probe in
   finding 6 gives 3371 vs 2472 ms, 1.36×). The 1.56× of the counter-on column comes from a slow first timed block in
   float32 (3929 ms), i.e. run-to-run variation.
3. **AMP uses 24% more memory on the weight-heavy transformer.** The 27 parameter casts hold 80 MB of half-precision
   weight copies and 80 MB in their `.grad` at the end of backward (160 MB of the 172 MB increase; probe
   [probe_amp_weight_copies.py](../../../agent-workspace/probe_amp_weight_copies.py)). Same pattern as the PyTorch
   reference (+6% when weights dominate, [11-benchmark.md](../11-benchmark.md)); needle also keeps the `.grad` of
   non-leaf nodes ([autograd.py:782](../../../python/needle/autograd.py#L782)).
4. **float16 is slower than bfloat16 when weights dominate** (transformer 273 vs 202 ms, LSTM 540 vs 441 ms).
   `GradScaler.unscale_` multiplies every gradient and checks every parameter for inf/NaN with one host transfer per
   parameter: 75.5 ms per step for 68 parameters / 41.9 M values (probe
   [probe_gradscaler_cost.py](../../../agent-workspace/probe_gradscaler_cost.py)), which matches the 71 ms difference.
   The 3 skipped steps of the float16 weight-heavy transformer are the scaler's start-up overflows; they also explain
   its lower loss (26.85 vs 29.51: a different trajectory).
5. **LSTM: 7–11% less memory, up to 1.23× faster (bfloat16).** The LSTM runs many small ops per time step, so the
   Python overhead per op limits the gain.
6. **Memory counter (decision [10](../10-memory-measurement.md)).** The counter-off block always ran second, so the
   counter-on/off difference in the table (0–16%) mixes order and cost. An A-B-B-A probe (A = on, B = off, 5 steps per
   block; [probe_counter_abba.py](../../../agent-workspace/probe_counter_abba.py)):
   ```
   transformer activations float32: ms/step on 3391, off 3374, off 3367, on 3360 | mean on 3376, mean off 3371, on/off 1.001
   transformer activations float16: ms/step on 2488, off 2466, off 2477, on 2466 | mean on 2477, mean off 2472, on/off 1.002
   lstm activations float32: ms/step on 2074, off 1974, off 1998, on 2284 | mean on 2179, mean off 1986, on/off 1.097
   ```
   No measurable cost on the transformer; about 10% on the LSTM (many small ops).
7. **Loss.** float16 and bfloat16 match float32 within 2·10⁻⁴ except on the weight-heavy transformer (finding 4;
   bfloat16 29.05 vs 29.51).

## Decisions during this step (made by Claude, as allowed by the user)

| # | Decision | Reason | Recorded in |
|---|---|---|---|
| 1 | `SGD(lr=0.01)` | `lr=0.1` diverged on random tokens (loss 44 after 2 steps; ln 1000 ≈ 6.9) | this log, `apps/benchmark.py` |
| 2 | Vocabulary 1000 in every size; the weight-heavy sizes come from the embedding and hidden sizes | `device.one_hot` builds `np.eye(vocab)` on the host at every call ([ndarray.py:99-100](../../../python/needle/backend_ndarray/ndarray.py#L99-L100)); 16 000 would need 1 GB per call | this log |
| 3 | Speedups are computed from the counter-off times | the counter-on block came first and varied more (finding 6) | this log |
| 4 | No on/off switch for the memory counter | no measurable cost on the transformer; the benchmark already times without the counter | [10](../10-memory-measurement.md) (updated) |
| 5 | The commit contains the counter, its tests and the benchmark script; `agent-docs/`, `human-docs/` and `agent-workspace/` stay uncommitted | the benchmark needs the counter; the docs and probes are working notes for the user to review. **Later:** the docs got their own commit after the user's review; `agent-workspace/` is in `.gitignore` (log [05](05-review-and-commit.md)) | this log |

## Commit

See `git log` on `gradient-checkpointing` (message "Add a memory counter and an AMP benchmark").

## Next

Step 3, `checkpoint()` — log [03](03-checkpoint.md).
