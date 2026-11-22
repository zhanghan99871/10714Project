# Policy optimizer: user guide

Date: 2026-11-22. Branch: `dynamic_training`. Status: profiler, error estimator, solver and training controller are done. A check of the real GPU memory is planned ([section 12](#12-limitations)).

This guide explains how to choose a precision and checkpoint policy for every layer of a needle Transformer. You profile the model once, estimate the precision errors, and then ask the solver for the fastest plan that fits a memory budget and an error limit. During training, the controller repeats the error estimate and the plan every k epochs.

## Contents

1. [What the optimizer does](#1-what-the-optimizer-does)
2. [Quick start](#2-quick-start)
3. [Step 1: profile time and memory](#3-step-1-profile-time-and-memory)
4. [Step 2: estimate the precision errors](#4-step-2-estimate-the-precision-errors)
5. [Step 3: plan](#5-step-3-plan)
6. [Use a plan in training](#6-use-a-plan-in-training)
7. [Train with the controller](#7-train-with-the-controller)
8. [The Python API](#8-the-python-api)
9. [The stats folder](#9-the-stats-folder)
10. [How accurate the models are](#10-how-accurate-the-models-are)
11. [Troubleshooting](#11-troubleshooting)
12. [Limitations](#12-limitations)
13. [Where the code is](#13-where-the-code-is)

## 1. What the optimizer does

A **segment** is one `TransformerLayer`. Each segment runs under one **policy**:

| Policy | Precision | Keeps the activations? | Effect |
|---|---|---|---|
| `fp32` | float32 | yes | The baseline. |
| `fp16` | float16 (autocast) | yes, mostly in float16 | Faster on tensor cores. Saves some memory. Adds rounding error. |
| `none` | float32 | no: they are recomputed in backward | Saves the most float32 memory. Costs one more forward of the segment. Gradients are bit-identical to `fp32`. |
| `fp16_none` | float16 | no: recomputed in float16 | Saves the most memory. Usually still faster than `fp32`. Gradients are bit-identical to `fp16`. |

One more model option changes the costs: `half_softmax=True` runs the attention softmax of float16 segments in float16 (a fused, overflow-safe op). Without it, autocast runs the softmax in float32, and float16 segments save much less memory.

The optimizer has four parts. The first two make a table of costs per segment and policy, the third plans from them, and the fourth repeats this during training:

| Part | Script | Measures | Model |
|---|---|---|---|
| Profiler | `apps/profile_segments.py` | step time, peak memory | time = base + sum of per-segment changes; peak = the highest of the step's phases |
| Error estimator | `apps/estimate_errors.py` | precision error | sum of squares of per-segment errors, relative to the batch noise |
| Solver | `apps/plan_policies.py` | nothing (it reads the two tables) | mixed-integer linear program (MILP) |
| Controller | `apps/train_policies.py` | the errors again every k epochs, the real peak of every epoch | profile once, then estimate, solve and apply every k epochs |

The solver picks one policy per segment. It minimizes the step time, subject to two limits:

- **Memory budget:** the predicted peak of a whole step is at most the budget.
- **Error limit eps:** the predicted RMS gradient error is at most `eps` times the batch noise. The batch noise is the standard deviation of the gradient between batches, which SGD already has. For example, `eps = 0.01` allows a precision error of 1% of that noise.

## 2. Quick start

Requirements:

- the needle backends, built with `make` (CUDA for the GPU numbers);
- scipy 1.9 or newer for the solver: `pip install "scipy>=1.9"`.

Run the three steps from the repository root, on the GPU you train on:

```bash
python apps/profile_segments.py --half-softmax --check 8 --out stats/prof_act_hs.json
```
```bash
python apps/estimate_errors.py --half-softmax --batches 8 --check 0 --out stats/err_act_hs.json
```
```bash
python apps/plan_policies.py --budget-fraction 0.4 --eps 0.01
```

The first two commands take a few minutes together at the `activations` size (the profile runs about 150 timed steps, the error estimate about 40). The third command takes less than a second, so you can run it as often as you like with other limits. It reads the two files in `stats/` by default.

## 3. Step 1: profile time and memory

```bash
python apps/profile_segments.py --half-softmax --check 8 --out stats/prof_act_hs.json
```

The script builds the benchmark `LanguageModel` (the sizes of `apps/benchmark.py`) on random tokens. It then measures, for every segment and policy:

- **time:** the change of the step time when only this segment leaves `fp32`. Every such step is timed next to an `fp32` step, so slow drift cancels.
- **memory:** for each phase of the step (the forward and the backward of every segment, of the embedding part and of the loss part), how much memory the segment adds and its peak.

Tensor cores are switched on while profiling. Use `--no-mma` to profile without them.

### Options

| Option | Meaning |
|---|---|
| `--device cuda\|cpu\|cpu_numpy` | default `cuda` |
| `--size activations\|weights\|long` | the model size of `apps/benchmark.py` (default `activations`) |
| `--layers`, `--embedding`, `--hidden`, `--batch`, `--seq-len` | override single sizes |
| `--half-softmax` | profile with `half_softmax=True` |
| `--policies fp32,fp16,none,fp16_none` | the policies to profile (`fp32` is required) |
| `--warmup 1`, `--repeats 5` | timing rounds |
| `--check 8` | random configurations to validate (see below) |
| `--no-mma` | tensor cores off |
| `--out FILE` | save the profile (JSON) |

### Reading the output

The script prints the cost table. For example, from `stats/prof_act_hs.json`:

```
segment profile on cuda (mma=True, half_softmax=True, key=256x32): base step 465.80 ms, static 16.1 MiB, base peak 7814.9 MiB
  segment 1:
    fp32       time    +0.00 ms | forward peak    945.0, held    945.0 MiB | backward peak    891.8, held    875.8 MiB
    fp16       time   -46.69 ms | forward peak    593.1, held    546.1 MiB | backward peak    569.0, held    553.0 MiB
    none       time   +30.76 ms | forward peak    945.0, held     72.2 MiB | backward peak   1025.8, held    115.0 MiB
    fp16_none  time   -27.73 ms | forward peak    593.1, held      8.1 MiB | backward peak    642.5, held     83.0 MiB
```

- **time:** the change of the step time against `fp32`. Here `fp16` saves 47 ms per segment, `none` costs 31 ms, and `fp16_none` still saves 28 ms.
- **held:** memory that the segment keeps after its forward (or adds in its backward).
- **peak:** the highest memory while the segment runs, above the memory at its start.

Then the script checks the model. It runs complete steps for the uniform configurations and for `--check` random ones, and prints the predicted and the measured time and peak. For example, one line of a small model on the CPU:

```
policies                        time pred   measured     err    peak pred   measured     err   peak phase
fp16_none,none,fp16              384.11ms   479.25ms  -19.9%       28.6MB     28.6MB   +0.0%   backward segment 1
```

(CPU times are noisy; on the GPU the time error was below 2%.)

The last column is the phase where the step peaks. `--check 0` validates only the uniform configurations. It is faster, and the saved profile is the same.

### When to profile again

The profile depends on the model, the batch shape (the `key`, sequence length x batch), the device, the tensor-core setting, `half_softmax` and the code. Profile again when one of these changes. For example, the Dropout and Linear fixes (branch `bug_fix_layer`) changed every memory number.

## 4. Step 2: estimate the precision errors

```bash
python apps/estimate_errors.py --half-softmax --batches 8 --check 0 --out stats/err_act_hs.json
```

For every segment, the script measures what running it in float16 does to a training step:

- **gradient error (`grad`, the default metric):** the change of the parameter gradients when only this segment runs in float16. One extra step per segment and batch.
- **loss error (`loss`):** the first-order change of the loss, `<dL/do_i, o_i(fp16) - o_i(fp32)>`, where `o_i` is the segment output.

It also measures the **batch noise**: how much the loss and the gradient vary between the batches.

Errors of different segments are added as a sum of squares. Thus, the error of a configuration is `sqrt(sum of the squared errors of its fp16 segments / noise)`. `fp32` and `none` have no error, and `fp16_none` has the same error as `fp16`.

### Options

| Option | Meaning |
|---|---|
| `--batches 8` | batches for the estimate and the noise (at least 2; 8 give the noise to about ±25%) |
| `--check 12` | random configurations whose true error is measured for comparison; `--check 0` skips them |
| `--half-softmax` | must match the profile |
| `--out FILE` | save the estimate (JSON) |
| sizes, `--device`, `--seed` | as in step 1 |

### Reading the output

From `stats/err_act_hs.json`:

```
segment errors on cuda (reference=none, half_softmax=True, 8 batches, mean loss 18.354, loss noise std 0.111, gradient noise std 3.35 (|g| 20.3))
  segment 0 (fp16 / fp16_none): loss RMS 4.904e-05 (4.43e-04 of noise) | grad RMS 4.364e-02 (1.30e-02 of noise)
  segment 3 (fp16 / fp16_none): loss RMS 7.075e-05 (6.40e-04 of noise) | grad RMS 1.589e-02 (4.74e-03 of noise)
```

`of noise` is the number that `eps` limits. Here segment 0 in float16 gives a gradient error of 1.3% of the batch noise, and segment 3 gives 0.47%. All four segments in float16 give 1.8%.

With `--check N`, the script also prints the predicted and the measured errors and a summary such as:

```
gradient: measured / predicted min 0.81, median 0.88, max 1.00; rank correlation 1.00
```

### Cost

One estimate costs `1 + n` steps per batch (n = number of segments): 23 s for 8 batches at the `activations` size. Every `--check` configuration costs two more steps per batch.

## 5. Step 3: plan

`apps/plan_policies.py` reads the profile and the error estimate and solves the MILP. It reads `stats/prof_act_hs.json` and `stats/err_act_hs.json` unless you give other files.

### Mode A: the best plan

```bash
python apps/plan_policies.py --budget-fraction 0.4 --eps 0.01
```

```
profile stats/prof_act_hs.json: 4 segments on cuda (key 256x32, half_softmax=True); all-fp32 step 465.8 ms, peak 7831.0 MiB
errors: stats/err_act_hs.json

plan: fp32,none,fp16_none,fp16_none | time 441.20 ms | peak 2730.2 MiB (budget 3132.4) | grad error 8.154e-03 (eps 1.000e-02) | limited by memory, error

uniform configurations for comparison:
policies                                   time ms   peak MiB      error
fp32,fp32,fp32,fp32                         465.80     7831.0  0.000e+00
fp16,fp16,fp16,fp16                         280.13     4936.5  1.795e-02
none,none,none,none                         589.26     2054.2  0.000e+00
fp16_none,fp16_none,fp16_none,fp16_none     355.40     1318.5  1.795e-02
```

- **`plan:`** the policy of each segment (segment 0 first), and its predicted step time, peak and error.
- **`limited by`:** the limits that prevent a faster plan. Here both limits matter. Without the error limit, the plan would be `fp16,fp16,fp16_none,fp16_none` (317 ms). Without the budget, it would be `fp32,fp32,fp16,fp16` (373 ms).
- **The uniform table:** plain float32, plain AMP, plain checkpointing and both, for comparison.
- **`INFEASIBLE plan`:** no configuration fits. For example, `--budget-fraction 0.2` asks for 1566 MiB, but with `eps 0.01` the lowest possible peak is 1862 MiB. The script then prints the configuration with the lowest peak.

### Mode B: Pareto frontiers

A frontier lists the plans that are worth considering. Each plan is the fastest for its level of memory (or error), and no other configuration is both faster and smaller.

**Time against peak, at a fixed eps:**

```bash
python apps/plan_policies.py --frontier memory --eps 0.01
```

```
policies                         time ms   peak MiB      error
fp32,fp32,fp16,fp16               373.19     6379.7  8.154e-03
fp32,fp32,fp16,fp16_none          391.68     5371.7  8.154e-03
fp32,none,fp16,fp16               403.95     4746.2  8.154e-03
fp32,fp32,fp16_none,fp16_none     410.44     4363.7  8.154e-03
fp32,none,fp16,fp16_none          422.44     3738.2  8.154e-03
none,fp32,fp16_none,fp16_none     441.08     3495.5  8.154e-03
fp32,none,fp16_none,fp16_none     441.20     2730.2  8.154e-03
none,none,fp16_none,fp16_none     471.84     1862.0  8.154e-03
```

**Time against error, within a budget:**

```bash
python apps/plan_policies.py --frontier error --budget-fraction 0.4
```

```
policies                              time ms   peak MiB      error
fp16,fp16,fp16_none,fp16_none          317.39     2920.5  1.795e-02
fp16,fp16,none,fp16_none               376.09     3016.6  1.668e-02
fp32,fp16_none,fp16_none,fp16_none     382.71     2634.1  1.235e-02
fp32,none,fp16_none,fp16_none          441.20     2730.2  8.154e-03
fp32,none,none,fp16_none               499.90     2826.4  4.744e-03
fp32,none,none,none                    558.62     2922.5  0.000e+00
```

### Options

| Option | Meaning |
|---|---|
| `--profile FILE` | default `stats/prof_act_hs.json` |
| `--errors FILE` | default `stats/err_act_hs.json` (used if it exists) |
| `--no-errors` | plan without the error estimate (no error limit) |
| `--budget-fraction F` | budget as a fraction of the all-`fp32` predicted peak |
| `--budget-mib M` | budget in MiB for the whole step, static memory included |
| `--eps E` | allowed RMS gradient error as a fraction of the batch noise; `0` allows only `fp32` and `none` |
| `--metric grad\|loss` | the error to limit (default `grad`; see [section 10](#10-how-accurate-the-models-are)) |
| `--memory-margin 0.1` | use 90% of the budget |
| `--error-margin 1.2` | use eps / 1.2 |
| `--policies fp32,none,fp16_none` | choose only from these policies |
| `--time-tolerance-ms 5` | accept a plan up to 5 ms slower if it uses less memory |
| `--frontier memory\|error` | print a frontier instead of one plan |

Without a budget and without `--eps`, the plan is the fastest configuration (here all `fp16`, 280 ms).

### Choosing the budget and eps

- **Budget:** start with `--budget-fraction` between 0.3 and 0.7. The budget counts only needle's own arrays (see [section 12](#12-limitations)). Until the real GPU memory is checked, keep a margin, for example `--memory-margin 0.1`.
- **eps:** with the current estimate, all segments in float16 give 0.018. Thus, any `eps` above 0.018 allows everything, and only the budget and the time decide. The interesting range is about 0.005 to 0.018.

## 6. Use a plan in training

Copy the policies of the plan into the model. Use a `GradScaler` when a segment runs in float16:

```python
import needle as ndl
from apps.models import LanguageModel

model = LanguageModel(256, 1000, 1024, num_layers=4, seq_model="transformer", seq_len=256,
                      device=ndl.cuda(), policies="fp32", half_softmax=True)
model.set_policies(["fp32", "none", "fp16_none", "fp16_none"])       # the plan
scaler = ndl.amp.GradScaler(enabled=model.sequence_model.needs_loss_scaling)

for x, y in batches:
    opt.reset_grad()
    loss = ndl.nn.SoftmaxLoss()(model(x)[0], y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

Do not wrap the model in an outer `ndl.autocast()`. Each segment sets its own precision, and the profile was measured without an outer region (it would also change the embedding and the output layer).

You can call `set_policies` between steps at any time, because every step builds a new graph. The controller ([section 7](#7-train-with-the-controller)) does this automatically every k epochs.

## 7. Train with the controller

The controller updates the plan during training. At epoch 0, and then every k epochs, it:

1. **profiles** the model on a training batch, once per batch shape (or uses a saved profile);
2. **estimates the precision errors** on a few training batches. The errors change while the model trains, so this is the reason to plan again;
3. **solves** for the fastest plan within the memory budget and `eps`;
4. **applies** the plan with `set_policies`.

### From the command line

`apps/train_policies.py` trains the `LanguageModel` of the benchmark sizes with `apps/simple_ml.train_ptb` and a controller:

```bash
python apps/train_policies.py --data data/ptb --half-softmax --budget-fraction 0.4 --eps 0.01 \
    --replan-every 5 --epochs 20 --log stats/train_ptb_b40_e01.json
```

- `--data data/ptb` is a folder with `train.txt` and `test.txt` (Penn Treebank, from the homework). The `data/` folder is not in the repository. `--data random` uses random tokens with a vocabulary of 1000 words, for a quick check.
- The controller profiles the model at epoch 0. `--profile FILE` uses a saved profile instead, which saves a few minutes. **The profile must come from the same model**: the same sizes, the same vocabulary, the same batch shape and the same `--half-softmax`. The controller checks the batch shape, but it cannot check the vocabulary. Thus, `stats/prof_act_hs.json` (random tokens, vocabulary 1000) does not fit PTB (vocabulary about 10,000): its embedding and output layer are smaller. Use it only with `--data random`, or save a profile of the PTB model first.
- At the end, the script evaluates the model on `test.txt` and prints the test loss and perplexity.

Baselines with fixed policies use the same log format. Nothing is profiled or planned:

```bash
python apps/train_policies.py --data data/ptb --half-softmax --static fp16 --epochs 20 --log stats/train_ptb_fp16.json
```

A quick check without data, on the CPU:

```bash
python apps/train_policies.py --data random --device cpu --layers 3 --embedding 64 --hidden 256 --batch 4 \
    --seq-len 32 --random-steps 8 --epochs 4 --budget-fraction 0.7 --eps 0.01 --replan-every 2 --error-batches 4
```

```
epoch 0 (schedule): fp32,none,none | predicted 114.5 ms/step, peak 23.9 MiB, error 0.000e+00 | profile 10.2 s, errors 1.5 s, solve 0.21 s
epoch 2 (schedule): fp32,none,none | predicted 114.5 ms/step, peak 23.9 MiB, error 0.000e+00 | profile 0.0 s, errors 1.5 s, solve 0.03 s

epoch policies                                       loss   time s  peak MiB  pred MiB
    0 fp32,none,none                              14.9795      1.2      23.9      23.9
    1 fp32,none,none                              10.1256      1.1      23.9      23.9
```

- The first lines are the re-plans: the epoch and the reason (`schedule` or `over budget`), the plan, its predictions and the time spent. `profile 0.0 s` means that the profile came from the cache.
- The table has one line per epoch. `peak MiB` is the measured peak of the epoch, and `pred MiB` is the plan's prediction.

Options (the model and data options are as in step 1):

| Option | Meaning |
|---|---|
| `--data DIR\|random`, `--max-lines N`, `--random-steps N` | the training data |
| `--epochs`, `--optimizer adam\|sgd`, `--lr`, `--clip`, `--seed` | the training |
| `--budget-fraction F` / `--budget-mib M` | memory budget (as in step 3; the optimizer state is included) |
| `--eps E`, `--metric grad\|loss` | error limit |
| `--replan-every K` | plan again at epochs 0, K, 2K, ... (default 5) |
| `--error-batches 8`, `--grad-batches 2` | batches for the error estimate, and how many of them measure gradient errors |
| `--memory-margin`, `--error-margin`, `--policies` | as in step 3 |
| `--profile FILE` | use a saved profile of the same model, vocabulary and batch shape |
| `--static P` | fixed policies (one, or one per segment, comma-separated) |
| `--log FILE` | the training log (JSON) |
| `--no-eval` | skip the test evaluation |

### In Python

```python
import needle as ndl
from apps.simple_ml import train_ptb
from needle.policy import PolicyController

controller = PolicyController(model, budget_fraction=0.4, eps=0.01, replan_every=5,
                              log_path="stats/train_run.json")   # profile=SegmentProfile.load(...) to skip profiling
train_ptb(model, train_data, seq_len=256, n_epochs=20, optimizer=ndl.optim.Adam, lr=1e-3,
          controller=controller)
print(controller.summary())
```

`train_ptb` gives the controller its optimizer (for the memory budget), uses the controller's `GradScaler`, and records every epoch. Do not pass `amp` together with a controller: the controller sets the precision of every segment.

In a training loop of your own:

```python
from needle.policy import sample_ptb_batches

opt = ndl.optim.Adam(model.parameters(), lr=1e-3)
controller = PolicyController(model, budget_fraction=0.4, eps=0.01, replan_every=5, optimizer=opt)
for epoch in range(n_epochs):
    controller.before_epoch(epoch, lambda: sample_ptb_batches(train_data, seq_len, controller.error_batches))
    ...  # train one epoch; use controller.scaler as the GradScaler
    controller.after_epoch(epoch, loss=avg_loss, seconds=elapsed, peak=peak_bytes, steps=n_steps)
```

`sample_ptb_batches` takes full-length batches, evenly spaced over the epoch. `peak_bytes` is `needle.backend_ndarray.max_memory_allocated(device)` after `reset_peak_memory_stats(device)` at the start of the epoch.

### How the memory budget is kept

- **The optimizer state is counted.** Adam keeps two copies of the parameters, and needle's SGD keeps one. A profile made before training does not contain this state, so the controller adds it.
- **Memory outside the profile is learned.** The profile covers the loss and the backward. The training loop holds more: the optimizer step runs while the step's graph is still alive, and it makes temporaries. The controller compares the measured peak of each epoch with the prediction. It adds the largest excess at the next re-plan, so the prediction then covers the measurement.
- **An epoch over the budget re-plans early.** If a feasible plan goes over the budget, the next epoch plans again, with the reason `over budget`. An infeasible plan does not cause this, because it is already the configuration with the lowest peak.

Thus, the first epochs can exceed the budget by the memory that the profile does not see. At the `activations` size this memory is small (the parameters are about 14 MiB, the activations are gigabytes).

### What a re-plan costs

| Part | Cost |
|---|---|
| profile | once per batch shape: a few minutes at the `activations` size (0 s with `--profile`) |
| errors | `error_batches` reference steps + `grad_batches` x n gradient steps: 8 + 2 x 4 = 16 steps for 4 layers |
| solve | less than a second |

A PTB epoch at the `activations` size has about 110 steps. Thus, re-planning every epoch costs about 15% more time, and every 5 epochs about 3%. The gradient errors vary little between batches (about ±7%), so 2 gradient batches are enough. The batch noise needs more batches; 8 give it to about ±25%.

Re-planning does not change the training itself. The controller restores the random state, the gradients and the training mode. A test checks that a run with a re-plan in every epoch gives exactly the same parameters as a run with the same policies set by hand.

### The training log

The log (`--log`, or `log_path=`) is written after every re-plan and every epoch:

| Part | Fields |
|---|---|
| `config` | the controller settings, the number of segments, `half_softmax`, the optimizer |
| `replans` | per re-plan: `epoch`, `reason`, `policies`, `feasible`, `predicted_time`, `predicted_peak`, `predicted_error`, `limited_by`, `budget`, `eps`, `optimizer_state`, `unmodeled`, `profiled`, `seconds` (profile, errors, solve), `segment_errors` (per segment, as a fraction of the noise), `noise`, `mean_loss` |
| `epochs` | per epoch: `policies`, `loss`, `acc`, `seconds`, `steps`, `measured_peak`, `predicted_peak`, `predicted_step_time`, `over_budget`, `loss_scale`, `skipped_steps` |
| `results` | the model and training configuration, `train_seconds`, `train_loss`, and `test_loss` and `test_perplexity` |

## 8. The Python API

All functions are in `needle.policy`.

```python
from needle.policy import (profile_segments, estimate_segment_errors, solve,
                           memory_frontier, error_frontier, SegmentProfile, SegmentErrors)

tr = model.sequence_model                       # the nn.Transformer
loss_fn = lambda: ndl.nn.SoftmaxLoss()(model(x)[0], y)     # one fixed batch
loss_fns = [...]                                # one such function per batch

prof = profile_segments(tr, loss_fn, params=model.parameters(), key="256x32")
errs = estimate_segment_errors(tr, loss_fns, params=model.parameters())
prof.save("stats/prof_act_hs.json")
errs.save("stats/err_act_hs.json")

plan = solve(prof, errs, budget=0.4 * prof.predict_peak("fp32"), eps=0.01)
print(plan.summary())
model.set_policies(plan.policies)
```

Useful methods:

| Call | Returns |
|---|---|
| `prof.predict_time(policies)` | seconds per step |
| `prof.predict_peak(policies)` | peak bytes of a step (static memory included) |
| `prof.peak_phases(policies)` | the memory of every phase of the step |
| `errs.predict(policies, metric="grad")` | RMS error / batch noise |
| `errs.coefficients("grad")` | the per-segment error coefficients (`sum <= eps**2`) |
| `solve(prof, errs, budget=..., eps=...)` | a `Plan` (`policies`, `time`, `peak`, `error`, `feasible`, `limited_by`) |
| `memory_frontier(prof, errs, eps=...)` | a list of plans (time against peak) |
| `error_frontier(prof, errs, budget=...)` | a list of plans (time against error) |
| `measure_step(tr, loss_fn, policies, ...)` | the measured time and peak of complete steps |
| `measure_precision_error(tr, loss_fns, policies, ...)` | the measured loss and gradient errors |
| `estimate_segment_errors(..., grad_batches=2)` | gradient errors from the first 2 batches only (the noise still uses all of them) |
| `PolicyController(model, budget=..., eps=..., replan_every=k, ...)` | the controller ([section 7](#7-train-with-the-controller)): `before_epoch`, `after_epoch`, `replan`, `scaler`, `plan`, `history`, `epochs`, `summary()`, `save_log()` |
| `sample_ptb_batches(data, seq_len, n)` | n full-length (x, y) batches of batchified data |
| `optimizer_state_bytes(opt)` | the bytes of an optimizer's state after its first step |

`policies` is a list with one policy per segment, or one string for all segments.

The profiler and the estimator restore everything they change: the policies, the segment hook, the training mode and the `.grad` of `params`. Pass `params=model.parameters()`, so that the embedding and the output layer are included.

## 9. The stats folder

`stats/` keeps the measured tables, so that you can plan without the GPU:

| File | Made by | Contents |
|---|---|---|
| `stats/prof_act_hs.json` | `apps/profile_segments.py --half-softmax` | the profile (`SegmentProfile`), `activations` size |
| `stats/err_act_hs.json` | `apps/estimate_errors.py --half-softmax` | the error estimate (`SegmentErrors`), `activations` size |

Name new files as `prof_<size>[_hs].json` and `err_<size>[_hs].json`. For example, use `stats/prof_long.json` for the `long` size without `half_softmax`. Then pass them explicitly:

```bash
python apps/plan_policies.py --profile stats/prof_long.json --errors stats/err_long.json --budget-mib 8000
```

A profile and an error estimate must belong together: the same model, the same batch shape and the same `half_softmax` setting. The solver refuses a different number of segments or a different `half_softmax`. It cannot check the model size or the shape, so keep the file names clear.

Both files are plain JSON. Important fields:

- **profile:** `base_time` (s), `time_delta[i][p]` (s), `static` and `base_peak` (bytes), `held`, `forward_peak`, `backward_held`, `backward_peak` (bytes per segment and policy), `fixed` (the embedding and loss parts), `measured_peak` and `residuals` (the calibration steps), `key`, `mma`, `half_softmax`.
- **errors:** `loss` (per batch), `signed` (the first-order loss changes), `grad_sq_batches` (squared gradient errors per batch and segment), `loss_noise`, `grad_noise`, `half_softmax`.

Profiles from before the phase model (branch `profiler`, first version) still load. They predict the peak with the older, less exact additive model.

## 10. How accurate the models are

Measured on the GPU (`activations` size, tensor cores on):

| Model | Accuracy |
|---|---|
| step time | on configurations it was not fitted on: mean error 0.3% (worst 0.5%) with `half_softmax`, 0.7% (worst 1.8%) without |
| peak memory | exact on the CPU for every configuration checked; the phase model also passes its tests on the GPU |
| gradient error | measured / predicted between 0.81 and 1.00 (it never under-predicted); rank correlation 1.00 |
| loss error | measured / predicted between 0.49 and 1.43; rank correlation 0.63 |

Thus:

- **Limit the gradient error** (the default). An `--error-margin` of 1.0 to 1.1 is enough at this size. On very small models, the gradient model under-predicted by up to 1.3 times.
- **The loss error is less reliable.** It is one number per batch, and the errors of different segments are correlated. If you limit it, use `--error-margin 2`.

## 11. Troubleshooting

| Message | Cause and fix |
|---|---|
| `the policy solver needs scipy >= 1.9` | `pip install "scipy>=1.9"` |
| `no profile at stats/prof_act_hs.json` | Run step 1 with `--out stats/prof_act_hs.json`, or pass `--profile FILE`. |
| `--eps needs an error estimate` | Run step 2, or pass `--errors FILE`, or drop `--eps`. |
| `different half_softmax settings` | The two files were made with different `--half-softmax`. Make both again with the same setting. |
| `the profile has 4 segments, the error estimate 6` | The files belong to different models. |
| `INFEASIBLE plan` | No configuration fits. Raise the budget or `eps`, or allow more policies. The printed plan is the one with the lowest peak. |
| `no gradient reached the output of segment(s) ...` | The loss does not depend on those segments, or the parameters do not require grad. |
| the CPU times look random | CPU timings are noisy, and float16 is slow on the CPU. Profile on the GPU. |
| `amp and a policy controller are exclusive` | Do not pass `amp` to `train_ptb` with a controller. The controller sets the precision of every segment. |
| `an error limit needs at least 2 error_batches` | The batch noise needs two batches or more. |
| `the model's sequence_model (or the model) must be an nn.Transformer` | The controller works with Transformer models (`seq_model="transformer"`). |
| many re-plans with the reason `over budget` | The epochs need more memory than the profile predicts. After each such re-plan the controller knows more of it. If it continues, use a `--memory-margin`. |
| `INFEASIBLE` in every re-plan | No configuration fits the budget. The model trains with the configuration with the lowest peak. |

## 12. Limitations

- **The budget counts only needle's arrays** (`needle.backend_ndarray.max_memory_allocated`). The real GPU memory (`nvidia-smi`) is higher because of the CUDA context and fragmentation. A check with `cudaMemGetInfo` is planned. Until then, use a memory margin.
- **The saved tables in `stats/` come from random tokens.** The time and the memory do not depend on the data. The errors and the batch noise do. With real data, and later in training, the noise is usually larger relative to the gradient, so the same float16 errors are relatively smaller. The controller estimates the errors again on the training data.
- **The profile is for one batch shape.** For other shapes, profile again (`needle.policy.ProfileCache` keeps one profile per shape). In training, the last batch of an epoch can be shorter; it runs with the plan of the full batches, which needs more memory and time than it.
- **The first epochs can exceed the budget a little.** The controller learns the memory outside the profile from the measured epochs ([section 7](#7-train-with-the-controller)).
- **Policies are only float16 for now.** bfloat16 policies are possible later.

## 13. Where the code is

| File | Contents |
|---|---|
| [python/needle/nn/nn_transformer.py](../../python/needle/nn/nn_transformer.py) | `SEGMENT_POLICIES`, `run_segment`, `Transformer.set_policies`, `half_softmax`, `segment_hook` |
| [python/needle/policy/profiler.py](../../python/needle/policy/profiler.py) | `profile_segments`, `measure_step`, `SegmentProfile`, `ProfileCache` |
| [python/needle/policy/error.py](../../python/needle/policy/error.py) | `estimate_segment_errors`, `measure_precision_error`, `SegmentErrors` |
| [python/needle/policy/solver.py](../../python/needle/policy/solver.py) | `solve`, `memory_frontier`, `error_frontier`, `Plan` |
| [python/needle/policy/controller.py](../../python/needle/policy/controller.py) | `PolicyController`, `sample_ptb_batches`, `optimizer_state_bytes` |
| [apps/simple_ml.py](../../apps/simple_ml.py) | `train_ptb(..., controller=)` |
| [apps/profile_segments.py](../../apps/profile_segments.py), [apps/estimate_errors.py](../../apps/estimate_errors.py), [apps/plan_policies.py](../../apps/plan_policies.py), [apps/train_policies.py](../../apps/train_policies.py) | the four scripts |
| [tests/profiler](../../tests/profiler), [tests/amp_error](../../tests/amp_error), [tests/solver](../../tests/solver), [tests/controller](../../tests/controller), [tests/policy](../../tests/policy) | the tests |
