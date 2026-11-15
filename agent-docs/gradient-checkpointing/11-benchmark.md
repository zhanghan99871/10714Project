# Decision 11 — Benchmark

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md) §6.
- **Working rule (user):** benchmarks for VRAM and speed may be added or edited without asking.

## Decision

1. **An AMP benchmark comes first**, before the checkpoint implementation (user):

| Part | Decision | Reason |
|---|---|---|
| Script | One script `apps/benchmark.py` with an AMP-dtype option; the checkpoint option is added later | AMP and checkpoint runs share model, data and measurement code |
| Models | `LanguageModel` with `transformer` and with `lstm` | AMP works on both |
| Runs | float32, float16 autocast (automatic `GradScaler`, [simple_ml.py:171-177](../../apps/simple_ml.py#L171-L177)), bfloat16 autocast, via `epoch_general_ptb(amp=...)` ([simple_ml.py:316-317](../../apps/simple_ml.py#L316-L317)) | The three modes of the AMP work |
| Sizes | Two per model: large activations (long sequences, large batch) and large weights (small batch) | The AMP saving depends on this ratio (PyTorch reference below) |
| Values | peak bytes (decision [10](10-memory-measurement.md)), time per training step, loss after the same number of steps, `GradScaler.skipped_steps` ([amp.py:176](../../python/needle/amp.py#L176)); **update 2026-11-14:** also the time per step with the memory counter replaced by a no-op, to show the counter's effect (decision 10, log [01](log/01-memory-counter.md)) | AMP changes rounding; report the loss difference to float32 |
| Timing | First steps are warm-up and not timed | The first step includes CUDA start-up |
| Data / device / results | Random tokens; cuda on GPU 1; results in `agent-docs/`, summary in `human-docs/` | As below |

2. **Checkpoint benchmark** (after the implementation):

| Part | Decision | Reason |
|---|---|---|
| Model | `LanguageModel(seq_model='transformer')`, one segment per `TransformerLayer` | Main use case; the layers are a `Sequential` |
| No LSTM | Not in the checkpoint benchmark | `LSTM.forward` loops over time steps with the layer loop inside ([nn_sequence.py:380-382](../../python/needle/nn/nn_sequence.py#L380-L382)); a segment per layer needs a change to that loop. Tests 1 and 8 (decision [09](09-tests.md)) cover correctness |
| Integration | Option `checkpoint=False` in `nn.Transformer.__init__`; when on, `forward` calls `checkpoint(layer, x)` per layer instead of `x = self.layers(x)` ([nn_transformer.py:387](../../python/needle/nn/nn_transformer.py#L387)); `LanguageModel` passes it through | ~5 lines; `simple_ml` unchanged |
| Data | Random tokens through `epoch_general_ptb` ([simple_ml.py:343-356](../../apps/simple_ml.py#L343-L356)) | No PTB data on this machine (`data/` does not exist); memory and time do not depend on token values |
| Runs | checkpoint off / on × float32 / float16 autocast / bfloat16 autocast; plus `autocast(release_inputs=True)` for decision [07](07-cast-of-segment-inputs.md) | Effect alone and with AMP |
| Values | peak bytes (`max_memory_allocated()`, decision [10](10-memory-measurement.md)); time per training step; loss off vs on (must be equal); after forward, number of segment inputs released by a cast (decision 07) | — |
| Device / size | cuda on GPU 1; sizes chosen so the largest run fits in 16 GB; the script prints them | GPU 1 is free; GPUs 0, 2–7 run other jobs |
| Location | `apps/benchmark.py` (checkpoint option added to the AMP benchmark script); results in `agent-docs/`, summary in `human-docs/` | Part of the project |

## Update 2026-11-14: as implemented (logs [02](log/02-amp-benchmark.md), [04](log/04-checkpoint-benchmark.md))

- `apps/benchmark.py --suite amp` (12 configurations; first run with the older flag `--all`) and `--suite checkpoint`
  (14 configurations). Claude's changes, allowed by the user: a new size `long` (seq_len 512) in the checkpoint suite;
  `SGD(lr=0.01)`; vocabulary 1000; speeds from the counter-off times; `release_inputs` through a replacement of
  `simple_ml._amp_context`; released segment inputs counted in one extra forward.
- Main results: AMP — transformer 1.35× faster with no memory saving, 1.24× memory when weights dominate; checkpoints —
  peak −71…73%, time +12…15%, sequence length 512 fits only with checkpoints.

## Comparison with PyTorch

The measurement mirrors PyTorch's `torch.cuda.max_memory_allocated()` / `reset_peak_memory_stats()` (decision 10).
A PyTorch reference for AMP memory, probe [agent-workspace/probe_torch_cast_memory.py](../../agent-workspace/probe_torch_cast_memory.py)
(8 × `Linear`+`ReLU`, forward + backward peak):
```
activation-heavy batch= 8192 width=2048 layers=8: peak float32    645 MB, autocast    420 MB
weight-heavy     batch=   16 width=4096 layers=8: peak float32    512 MB, autocast    544 MB
```
The AMP saving depends on the ratio of activations to weights (float16 weight copies saved for backward:
[agent-workspace/probe_torch_amp_weight_copies.py](../../agent-workspace/probe_torch_amp_weight_copies.py)).
