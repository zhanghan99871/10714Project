# Log 02: The AMP benchmark (step 2)

Date: 2026-11-14. Branch: `gradient-checkpointing`. Status: done, run and committed.
The full version is in [agent-docs/gradient-checkpointing/log/02-amp-benchmark.md](../../../agent-docs/gradient-checkpointing/log/02-amp-benchmark.md).

## What we did

- `apps/benchmark.py` runs one configuration in each process. `--all` runs 12 configurations: 2 models, 2 sizes and 3 dtypes.
- Each run trains a language model on random tokens on GPU 1: 2 warm-up steps, 10 steps with the memory counter, and 10 steps without it.

## Results

| Model | Size | dtype | Peak memory compared with float32 | Speed compared with float32 |
|---|---|---|---|---|
| Transformer | Large activations | float16 | 1.01 | 1.35 |
| Transformer | Large activations | bfloat16 | 1.01 | 1.35 |
| Transformer | Large weights | float16 | 1.24 | 0.90 |
| Transformer | Large weights | bfloat16 | 1.24 | 1.22 |
| LSTM | Large activations | float16 | 0.89 | 1.12 |
| LSTM | Large activations | bfloat16 | 0.89 | 1.23 |
| LSTM | Large weights | float16 | 0.93 | 0.84 |
| LSTM | Large weights | bfloat16 | 0.93 | 1.02 |

## Findings

1. **AMP does not decrease the memory of the transformer.** `nn.Linear` adds the float32 bias in a separate op. This op changes the float16 result back to float32. Thus, most activations stay float32.
2. **AMP makes the transformer with large activations about 1.35 times faster.**
3. **With large weights, AMP uses 24% more memory.** Autocast keeps half-precision copies of the weights and their gradients until the end of the step (160 MB).
4. **With large weights, float16 is slower than bfloat16.** `GradScaler` checks each gradient, and this takes 75 ms in each step. bfloat16 does not need `GradScaler`.
5. **The LSTM gets 7–11% less memory and is up to 1.23 times faster.** It has many small ops, so the Python time limits the gain.
6. **The memory counter has no measurable cost on the transformer.** On the LSTM, it costs about 10%. Thus, we do not add an on/off switch.
7. **The loss of float16 and bfloat16 is almost the same as float32,** except on the transformer with large weights.

## Decisions in this step (made by Claude)

1. The learning rate is 0.01, because 0.1 made the training diverge.
2. The vocabulary is 1000 in all sizes, because each step builds a `vocab × vocab` matrix on the host.
3. The speeds come from the steps without the counter, because those times vary less.
4. The memory counter gets no on/off switch.
5. The commit contains the counter, its tests and the benchmark. The docs and the test scripts in `agent-workspace/` are not committed. Later, the docs got their own commit, and `agent-workspace/` is in `.gitignore` (log 05).
