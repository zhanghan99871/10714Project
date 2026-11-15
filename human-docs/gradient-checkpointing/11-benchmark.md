# Decision 11: The benchmark

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/11-benchmark.md](../../agent-docs/gradient-checkpointing/11-benchmark.md).

## Decision

Working rule: Claude can add or change benchmarks for VRAM and speed without asking.

1. We write and run a benchmark for AMP first, before we write the checkpoint code:
   - **Script:** `apps/benchmark.py`, with an option for the AMP dtype. The checkpoint option comes later.
   - **Models:** `LanguageModel` with `transformer`, and with `lstm`.
   - **Runs:** float32, float16 autocast with `GradScaler`, and bfloat16 autocast, through `epoch_general_ptb(amp=...)`.
   - **Sizes:** for each model, one size with large activations and one with large weights.
   - **Values:** the peak memory, the time per training step, the loss after the same number of steps, and the number of steps that `GradScaler` skipped.
   - **Update (2026-11-14):** the benchmark also measures the time per step without the memory counter. This shows the effect of the counter (decision 10).
   - **Timing:** the first steps are warm-up, and the benchmark does not time them.
   - **Data and device:** random tokens, cuda on GPU 1.
2. After the checkpoint code, a checkpoint benchmark follows:
   - **Model:** `LanguageModel(seq_model='transformer')`, with one segment for each `TransformerLayer`.
   - **No LSTM:** the loop in `LSTM.forward` does not give one segment per layer. The tests check its correctness.
   - **Integration:** `nn.Transformer` gets the option `checkpoint=False`. When the option is on, `forward` calls `checkpoint(layer, x)` for each layer.
   - **Data:** random tokens, because the PTB data is not on this machine.
   - **Runs:** checkpoint off and on, each with float32, float16 and bfloat16. One more run uses `autocast(release_inputs=True)` for decision 07.
   - **Values:** the peak memory, the time per training step, the loss (it must be the same with checkpoint off and on), and the number of segment inputs that a cast released.
   - **Device:** cuda on GPU 1.
   - **Location:** the same script, `apps/benchmark.py`, with a checkpoint option.

## Update: as implemented (logs 02 and 04)

- `apps/benchmark.py --suite amp` runs 12 configurations, and `--suite checkpoint` runs 14.
- Claude added a size `long` (sequence length 512), set the learning rate to 0.01, and set the vocabulary to 1000.
- **AMP:** the transformer is 1.35 times faster, but its memory does not decrease. With large weights, AMP uses 24% more memory.
- **Checkpoints:** the peak memory decreases by 71–73%, and each step takes 12–15% more time. A sequence length of 512 fits only with checkpoints.

## Comparison with PyTorch

- The memory functions have the same names as in PyTorch.
- A PyTorch test showed that the memory saving of AMP depends on the size of the activations compared with the weights. With large activations, AMP used 35% less memory. With large weights, it used 6% more.
