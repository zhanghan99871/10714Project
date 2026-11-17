# Gradient checkpointing — overall plan

- Owner: xiachu (gradient checkpointing). AMP / bf16 / WMMA were done earlier by a colleague (PR #3–#9).
- Date: 2026-11-14. Base commit: `83ae951`.
- Status: steps 1–4 done and committed: steps 1–2 in `d011880` (logs [01](log/01-memory-counter.md), [02](log/02-amp-benchmark.md)), steps 3–4 in `28fc334` (logs [03](log/03-checkpoint.md), [04](log/04-checkpoint-benchmark.md)). Review and commits: log [05](log/05-review-and-commit.md).
- Decisions: [01-api-form.md](01-api-form.md), [02-segment-membership.md](02-segment-membership.md),
  [03-segment-outputs.md](03-segment-outputs.md), [04-nested-segments.md](04-nested-segments.md),
  [05-backward-release-scope.md](05-backward-release-scope.md), [06-tensortuple.md](06-tensortuple.md),
  [07-cast-of-segment-inputs.md](07-cast-of-segment-inputs.md),
  [08-special-modes.md](08-special-modes.md), [09-tests.md](09-tests.md),
  [10-memory-measurement.md](10-memory-measurement.md),
  [11-benchmark.md](11-benchmark.md).
- Working rule (user): tests may be added, edited or removed without asking.
- Working rule (user): benchmarks for VRAM and speed may be added or edited without asking.
- Working rule (user): work in the branch `gradient-checkpointing`; `main` stays at `83ae951`.
- Working rule (user): start an implementation step only when the user says "start".
- Line references: in this plan and decisions 01–11 they refer to `83ae951` (before the counter and checkpoint changes; `ndarray.py`, `autograd.py` lines have moved since). The logs and the update section of decision 05 refer to `28fc334`.
- Process log: [log/](log/) — one file per step of the order of work, with the decisions made during the step.

## 1. The plan (proposed by the user)

1. `checkpoint(fn, *args)` runs `fn` as one **segment**.
2. At the end of the segment, release the `cached_data` of the segment's activations. Keep the segment inputs and outputs (the "checkpoints").
3. During backward, `realize_cached_data()` recomputes a released activation when a gradient needs it.
4. After backward has processed a node, release that node again.

## 2. Why the plan fits the existing engine

### 2.1 Recompute already exists
`Value.realize_cached_data()` recomputes a released node from its inputs, recursively, and caches the result
([python/needle/autograd.py:315-333](../../python/needle/autograd.py#L315-L333)):

```python
def realize_cached_data(self):
    if self.cached_data is not None:
        return self.cached_data
    if self.op is None:
        raise RuntimeError("Cannot recompute a leaf tensor whose data was released.")
    data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
    self.cached_data = data
    ...
```

`Value.release_cached_data()` sets `cached_data = None`, keeps `_shape/_dtype/_device`, and refuses leaves
([autograd.py:335-346](../../python/needle/autograd.py#L335-L346)). Covered by
[tests/design/test_metadata.py](../../tests/design/test_metadata.py). The AMP work already uses this: `ops.Cast`
releases its non-leaf input by default ([ops_mathematic.py:50-72](../../python/needle/ops/ops_mathematic.py#L50-L72)).

### 2.2 Step 3 needs no new code
Gradients read `node` and `node.inputs`, e.g. `EWiseMul`
([ops_mathematic.py:117-119](../../python/needle/ops/ops_mathematic.py#L117-L119)):

```python
def gradient(self, out_grad, node):
    lhs, rhs = node.inputs
    return out_grad * rhs, out_grad * lhs
```

`out_grad * rhs` is a new op; `Tensor.make_from_op` realizes it, which calls `rhs.realize_cached_data()` → recompute.

### 2.3 Step 4 is safe
`_compute_gradient_of_variables` visits nodes in reverse topological order and processes each node once
([autograd.py:716-743](../../python/needle/autograd.py#L716-L743)). When node `k` is processed, every consumer of `k`
was processed before, and `k`'s own gradient has just been computed. No later step of the same backward pass reads
`k`. (With `create_graph=True` the gradient graph references forward nodes; a released one is recomputed on demand.)

### 2.4 Probe: release + backward works today
[agent-workspace/probe_release_backward.py](../../agent-workspace/probe_release_backward.py): 3 × (`matmul → tanh`),
release the 5 interior nodes, keep the output, run backward.

```
grads equal: True
matmul forwards during backward: kept = 6  released = 8
interior nodes still released after backward: [False, False, False, False, True, False]
TensorTuple release: Cannot release an uninitialized tensor.
```

- Gradients bit-identical.
- Only 2 extra matmul forwards; the last matmul output is never recomputed because `Tanh.gradient` reads only its
  output. The recompute is lazy: only nodes on a path that some gradient reads are recomputed.
- After backward the 4 recomputed nodes still hold data → step 4 is required.

## 3. Comparison with PyTorch: how activations are recomputed

Reference: PyTorch 2.10.0, `/home/xiachu/.conda/envs/vda/lib/python3.12/site-packages/torch/utils/checkpoint.py`.

### 3.1 PyTorch runs `fn` again, in both modes
- `use_reentrant=True`: forward runs `fn` under `torch.no_grad()` (L267-268); backward runs `fn` again under
  `enable_grad` on detached inputs and calls `torch.autograd.backward` on the new outputs (L307-325).
- `use_reentrant=False` (recommended): forward records the graph under `_checkpoint_hook` (L1607); the first unpack
  in backward calls `frame.recompute_fn` (L1164-1173), which calls `fn(*args, **kwargs)` (L1591).
- Probe [agent-workspace/probe_nonreentrant_rerun.py](../../agent-workspace/probe_nonreentrant_rerun.py):
  ```
  calls after forward : 1 graph task ids: [-1]
  calls after backward: 2 graph task ids: [-1, 0]
  ```
  The second call has graph task id 0 → it runs inside backward.

Because `fn` runs twice, PyTorch must make the second run equal the first:
- RNG: `preserve_rng_state=True` stashes and restores CPU/CUDA RNG state (L1576-1585).
- Autocast: saved and re-entered for the recompute (L1587-1591).
- Anything else is the user's problem. Docstring (L376-381): *"If the function invocation during the backward pass
  differs from the forward pass, e.g., due to a global variable, the checkpointed version may not be equivalent…"*
- `determinism_check="default"` compares only shape/dtype/device of recomputed tensors, not values (L1063-1069).
- Side effects run twice. Probe [agent-workspace/probe_batchnorm_torch.py](../../agent-workspace/probe_batchnorm_torch.py):
  ```
  no checkpoint          num_batches_tracked: after forward=1, after backward=1; running_mean changed in backward: False
  use_reentrant=False    num_batches_tracked: after forward=1, after backward=2; running_mean changed in backward: True
  use_reentrant=True     num_batches_tracked: after forward=1, after backward=2; running_mean changed in backward: True
  ```

### 3.2 Our plan replays the recorded ops, not `fn`
Recompute calls `op.compute()` on the recorded nodes. Python code in `fn` / `Module.forward` does not run again.
Consequences:
- **Dropout**: the mask is a leaf constant ([nn_basic.py:236-237](../../python/needle/nn/nn_basic.py#L236-L237)).
  Leaves are never released, so the replay uses the same mask. No RNG handling needed.
- **BatchNorm**: running stats are updated by a Python assignment with `.data`
  ([nn_basic.py:176-177](../../python/needle/nn/nn_basic.py#L176-L177)); the result is a detached constant
  ([autograd.py:501-502](../../python/needle/autograd.py#L501-L502)) and not part of the graph. The replay does not
  update them again. Probe [agent-workspace/probe_batchnorm_needle.py](../../agent-workspace/probe_batchnorm_needle.py):
  `released nodes: 21`, `running_mean changed in backward: False`.
- **Autocast**: the `Cast` nodes inserted by autocast are in the graph
  ([autograd.py:142-145](../../python/needle/autograd.py#L142-L145), [autograd.py:494-497](../../python/needle/autograd.py#L494-L497)),
  so the replay uses the forward dtypes. No autocast state handling needed.
- **Early stop**: PyTorch non-reentrant stops recomputing once all needed saved tensors exist (docstring L396-400).
  Our recompute is lazy per node (§2.4), which gives the same effect without extra code.

Costs of our approach versus PyTorch:
- The graph structure of the segment (Python `Tensor` objects, `op` objects, `inputs` lists) stays alive in forward.
  PyTorch non-reentrant also keeps its graph (with empty holders); reentrant keeps only the inputs.
- **Leaf constants inside the segment stay in memory**, e.g. a Dropout mask has the shape and dtype of its activation.
  PyTorch regenerates the mask from the restored RNG state instead of storing it.

## 4. Decisions

| # | Topic | Decision | File |
|---|-------|----------|------|
| 01 | API form, location, name | `checkpoint(fn, *args)` first; module wrapper and `checkpoint_sequential` later; in `autograd.py`, exported as `ndl.checkpoint` | [01-api-form.md](01-api-form.md) |
| 02 | Which nodes belong to a segment | Record every node that an op makes while the segment runs (hook in `Value._init`), in a plain list (strong references) dropped at the segment end | [02-segment-membership.md](02-segment-membership.md) |
| 03 | How `checkpoint()` finds segment outputs | Search the return value through nested tuples and lists; keep every `Tensor` found; dicts not supported yet (stated in the docstring) | [03-segment-outputs.md](03-segment-outputs.md) |
| 04 | Nested segments | Not supported (stated in the docstring, no error) | [04-nested-segments.md](04-nested-segments.md) |
| 05 | Which nodes backward releases after processing | Only interior nodes of checkpoint segments; interior nodes carry a flag; their adjoints are dropped too (update); the read-after-step limit is documented in the `checkpoint()` docstring | [05-backward-release-scope.md](05-backward-release-scope.md) |
| 06 | `TensorTuple` nodes | Released like `Tensor` nodes; guard removed; record hook in `Value._init`; outputs may be `TensorTuple` | [06-tensortuple.md](06-tensortuple.md) |
| 07 | Casts that release a segment input | No protection in the first version; tests measure recompute and VRAM; protect later if needed | [07-cast-of-segment-inputs.md](07-cast-of-segment-inputs.md) |
| 08 | `no_grad`, no grad, `LAZY_MODE` | No code for `no_grad` and no grad (`fn` only runs); `LAZY_MODE` not supported (docstring) | [08-special-modes.md](08-special-modes.md) |
| 09 | Correctness tests | 12 tests in `tests/checkpoint/test_checkpoint.py` on numpy/cpu/cuda; tests may change without asking | [09-tests.md](09-tests.md) |
| 10 | Memory measurement | Python counter in `NDArray.make`, per device; `memory_allocated()`, `max_memory_allocated()`, `reset_peak_memory_stats()` | [10-memory-measurement.md](10-memory-measurement.md) |
| 11 | Benchmark | AMP benchmark first (Transformer and LSTM LM, 3 dtypes, 2 sizes); then the checkpoint benchmark on the Transformer LM (`checkpoint=False` option), cuda, GPU 1 | [11-benchmark.md](11-benchmark.md) |

## 5. Known problems and interactions

- **`TensorTuple` cannot be released** (resolved by decision [06](06-tensortuple.md): the guard is removed). `_update_metadata()` runs only for `Tensor`
  ([autograd.py:330](../../python/needle/autograd.py#L330), [autograd.py:388](../../python/needle/autograd.py#L388)), so
  `_shape` stays `None` and `release_cached_data()` raises "Cannot release an uninitialized tensor."
  `ops.split` returns a `TensorTuple`; RNN/LSTM use it ([nn_sequence.py:180](../../python/needle/nn/nn_sequence.py#L180),
  [nn_sequence.py:312](../../python/needle/nn/nn_sequence.py#L312)).
- **Recompute after the optimizer step uses the new weights.** Optimizers replace leaf data
  (`param.data -= ...`, [optim.py:38](../../python/needle/optim.py#L38); setter [autograd.py:523-531](../../python/needle/autograd.py#L523-L531)).
  A released node read after `opt.step()` is recomputed with the new weights. Probe
  [agent-workspace/probe_recompute_after_step.py](../../agent-workspace/probe_recompute_after_step.py):
  `out before step: [3. 3.]`, `out read after step: [2.4 2.4]`. Decision 05 limits this to interior nodes.
- **Explicit casts inside a segment release their input** (decision [07](07-cast-of-segment-inputs.md): no protection in the first version; measure). `ops.cast(x, ...)` with the default `release_input=True`
  releases a non-leaf `x` ([ops_mathematic.py:50-53](../../python/needle/ops/ops_mathematic.py#L50-L53)). If `x` is a
  segment input, the checkpoint is gone and the recompute recurses into the earlier graph. Correct, but costs more.
  Autocast's own casts keep their input by default (`release_inputs=False`, [amp.py:81](../../python/needle/amp.py#L81)).
- **Autocast parameter-cast cache across segments.** A cast of a parameter made in segment 1 is cached
  ([autograd.py:113-122](../../python/needle/autograd.py#L113-L122)) and reused in segment 2. Under decision 02 it is
  recorded by segment 1, released at its end, recomputed when segment 2 uses it, and then stays cached.
- **The previous step's graph stays alive during the next forward** (independent of checkpointing). Probe
  [agent-workspace/probe_needle_graph_lifetime.py](../../agent-workspace/probe_needle_graph_lifetime.py):
  `at start of next forward, previous hidden buffer alive: True`. See decision 05, §"Not in scope".

## 6. Open questions (not decided)

None. All design decisions are recorded.

## 6a. Order of work

1. Memory counter (decision [10](10-memory-measurement.md)) and its tests — the AMP benchmark needs it. **Done, committed** (`d011880`, log [01](log/01-memory-counter.md)).
2. AMP benchmark: write, run, record the results. **Done, committed** (`d011880`, log [02](log/02-amp-benchmark.md)).
3. `checkpoint()` (decisions 01–08) and its tests (decision [09](09-tests.md)). **Done, committed** (`28fc334`, log [03](log/03-checkpoint.md)).
4. Checkpoint benchmark (decision [11](11-benchmark.md)) and the decision-07 measurement. **Done, committed** (`28fc334`, log [04](log/04-checkpoint-benchmark.md)).

Removed (answered): checkpoint benchmark → [11](11-benchmark.md); memory measurement → [10](10-memory-measurement.md); correctness test scope → [09](09-tests.md); `no_grad` / no grad / `LAZY_MODE` → [08](08-special-modes.md); casts that release a segment input → [07](07-cast-of-segment-inputs.md); `TensorTuple` handling → [06](06-tensortuple.md); nested segments → [04](04-nested-segments.md); interior-node flag → [05](05-backward-release-scope.md); record storage → [02](02-segment-membership.md); API form, code location and public name → [01](01-api-form.md); outputs of `fn` → [03](03-segment-outputs.md); where to document
the read-after-step limit → [05](05-backward-release-scope.md), "Consequences and limits".

## 7. Environment

- Conda env `needle`: Python 3.11.18, numpy 2.4.6, ml_dtypes 0.6.0, pybind11 3.1.0, pytest 9.2.0, CMake 3.31.10
  (CMake ≥ 3.27 is needed for policy `CMP0146`, [CMakeLists.txt:3](../../CMakeLists.txt#L3)).
- Build: `PATH=/home/xiachu/.conda/envs/needle/bin:$PATH make` (CUDA 12.0, `sm_86`, RTX A4000).
- Baseline on `83ae951`: `python -m pytest tests -q` → **828 passed** in 30.61 s
  ([agent-workspace/pytest_baseline.log](../../agent-workspace/pytest_baseline.log)).
- PyTorch for reference only: env `vda` (torch 2.10.0+cu128).
