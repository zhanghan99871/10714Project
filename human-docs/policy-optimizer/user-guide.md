# Policy optimizer: user guide

Date: 2026-11-22. Branch: `solver`. Status: profiler, error estimator and solver are done. The training controller (re-planning every k epochs during training) is not written yet.

This guide explains how to choose a precision and checkpoint policy for every layer of a needle Transformer. You profile the model once, estimate the precision errors, and then ask the solver for the fastest plan that fits a memory budget and an error limit.

## Contents

1. [What the optimizer does](#1-what-the-optimizer-does)
2. [Quick start](#2-quick-start)
3. [Step 1: profile time and memory](#3-step-1-profile-time-and-memory)
4. [Step 2: estimate the precision errors](#4-step-2-estimate-the-precision-errors)
5. [Step 3: plan](#5-step-3-plan)
6. [Use a plan in training](#6-use-a-plan-in-training)
7. [The Python API](#7-the-python-api)
8. [The stats folder](#8-the-stats-folder)
9. [How accurate the models are](#9-how-accurate-the-models-are)
10. [Troubleshooting](#10-troubleshooting)
11. [Limitations](#11-limitations)
12. [Where the code is](#12-where-the-code-is)

## 1. What the optimizer does

A **segment** is one `TransformerLayer`. Each segment runs under one **policy**:

| Policy | Precision | Keeps the activations? | Effect |
|---|---|---|---|
| `fp32` | float32 | yes | The baseline. |
| `fp16` | float16 (autocast) | yes, mostly in float16 | Faster on tensor cores. Saves some memory. Adds rounding error. |
| `none` | float32 | no: they are recomputed in backward | Saves the most float32 memory. Costs one more forward of the segment. Gradients are bit-identical to `fp32`. |
| `fp16_none` | float16 | no: recomputed in float16 | Saves the most memory. Usually still faster than `fp32`. Gradients are bit-identical to `fp16`. |

One more model option changes the costs: `half_softmax=True` runs the attention softmax of float16 segments in float16 (a fused, overflow-safe op). Without it, autocast runs the softmax in float32, and float16 segments save much less memory.

The optimizer has three parts. Each part makes a table of costs per segment and policy:

| Part | Script | Measures | Model |
|---|---|---|---|
| Profiler | `apps/profile_segments.py` | step time, peak memory | time = base + sum of per-segment changes; peak = the highest of the step's phases |
| Error estimator | `apps/estimate_errors.py` | precision error | sum of squares of per-segment errors, relative to the batch noise |
| Solver | `apps/plan_policies.py` | nothing (it reads the two tables) | mixed-integer linear program (MILP) |

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
| `--metric grad\|loss` | the error to limit (default `grad`; see [section 9](#9-how-accurate-the-models-are)) |
| `--memory-margin 0.1` | use 90% of the budget |
| `--error-margin 1.2` | use eps / 1.2 |
| `--policies fp32,none,fp16_none` | choose only from these policies |
| `--time-tolerance-ms 5` | accept a plan up to 5 ms slower if it uses less memory |
| `--frontier memory\|error` | print a frontier instead of one plan |

Without a budget and without `--eps`, the plan is the fastest configuration (here all `fp16`, 280 ms).

### Choosing the budget and eps

- **Budget:** start with `--budget-fraction` between 0.3 and 0.7. The budget counts only needle's own arrays (see [section 11](#11-limitations)). Until the real GPU memory is checked, keep a margin, for example `--memory-margin 0.1`.
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

You can call `set_policies` between steps at any time, because every step builds a new graph. The planned training controller will do this automatically every k epochs.

## 7. The Python API

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

`policies` is a list with one policy per segment, or one string for all segments.

The profiler and the estimator restore everything they change: the policies, the segment hook, the training mode and the `.grad` of `params`. Pass `params=model.parameters()`, so that the embedding and the output layer are included.

## 8. The stats folder

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

## 9. How accurate the models are

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

## 10. Troubleshooting

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

## 11. Limitations

- **The budget counts only needle's arrays** (`needle.backend_ndarray.max_memory_allocated`). The real GPU memory (`nvidia-smi`) is higher because of the CUDA context and fragmentation. A check with `cudaMemGetInfo` is planned. Until then, use a memory margin.
- **The tables come from random tokens.** The time and the memory do not depend on the data. The errors and the batch noise do. With real data, and later in training, the noise is usually larger relative to the gradient, so the same float16 errors are relatively smaller. Estimate the errors again during training (the planned controller does this).
- **The profile is for one batch shape.** For other shapes, profile again (`needle.policy.ProfileCache` keeps one profile per shape).
- **Policies are only float16 for now.** bfloat16 policies are possible later.
- **The training controller is not written yet.** You set the plan by hand (see [section 6](#6-use-a-plan-in-training)).

## 12. Where the code is

| File | Contents |
|---|---|
| [python/needle/nn/nn_transformer.py](../../python/needle/nn/nn_transformer.py) | `SEGMENT_POLICIES`, `run_segment`, `Transformer.set_policies`, `half_softmax`, `segment_hook` |
| [python/needle/policy/profiler.py](../../python/needle/policy/profiler.py) | `profile_segments`, `measure_step`, `SegmentProfile`, `ProfileCache` |
| [python/needle/policy/error.py](../../python/needle/policy/error.py) | `estimate_segment_errors`, `measure_precision_error`, `SegmentErrors` |
| [python/needle/policy/solver.py](../../python/needle/policy/solver.py) | `solve`, `memory_frontier`, `error_frontier`, `Plan` |
| [apps/profile_segments.py](../../apps/profile_segments.py), [apps/estimate_errors.py](../../apps/estimate_errors.py), [apps/plan_policies.py](../../apps/plan_policies.py) | the three scripts |
| [tests/profiler](../../tests/profiler), [tests/amp_error](../../tests/amp_error), [tests/solver](../../tests/solver), [tests/policy](../../tests/policy) | the tests |
