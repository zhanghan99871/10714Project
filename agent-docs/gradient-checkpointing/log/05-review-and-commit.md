# Log 05 — Review of `autograd.py` and the commits

- Date: 2026-11-15. Branch: `gradient-checkpointing`. Status: **done**.
- Commits: `28fc334` (code: steps 3–4), then the docs commit that contains this log.
- Line numbers refer to `28fc334`.

## Review of `autograd.py`

The user read the other files and asked for a walkthrough of [autograd.py](../../../python/needle/autograd.py) only.
The user replaced the comment above the flag in `Value._init`
([autograd.py:397-398](../../../python/needle/autograd.py#L397-L398)); `28fc334` keeps the user's comment.

### Finding: a nested `checkpoint()` call raises a `TypeError`

Decision [04](../04-nested-segments.md): not supported, documented in the docstring, no error of our own. A nested call
fails anyway: the inner call sets `_CHECKPOINT.record` to `None` when it ends
([autograd.py:715](../../../python/needle/autograd.py#L715)), and the outer call then loops over `None`
([autograd.py:717](../../../python/needle/autograd.py#L717)). Probe
[probe_nested_checkpoint.py](../../../agent-workspace/probe_nested_checkpoint.py):
```
TypeError 'NoneType' object is not iterable
```
No code change.

### Measurement: memory of interior `.grad` and of the partial-adjoint lists (decision 05 update)

Probe [probe_adjoint_memory.py](../../../agent-workspace/probe_adjoint_memory.py)
(log [probe_adjoint_memory.log](../../../agent-workspace/probe_adjoint_memory.log)): GPU 1, benchmark transformer
(activations size, float32), one forward + backward, no optimizer step; a copy of `_compute_gradient_of_variables`
with the two drops switchable.

| Variant | Peak | After backward |
|---|---|---|
| no checkpoints | 10 510.9 MB | 9330.7 MB |
| release the data only | 6265.9 MB | 4943.7 MB |
| data + drop `.grad` | 5961.9 MB | 1092.7 MB |
| data + drop the lists | 5849.9 MB | 4943.7 MB |
| data + drop both (`28fc334`) | **3014.2 MB** | 1092.7 MB |

Loss 18.453640 in all variants. At the end of backward, in "data only", the 540 interior tensors hold 3403.1 MB in
buffers shared by `.grad` and the lists, 480.0 MB only in `.grad`, and 512.0 MB only in the lists.

- **Shared buffers:** 500 of the 540 interior tensors get exactly one partial adjoint. `_sum_adjoints` then returns
  `sum_node_list([g])` ([autograd.py:763](../../../python/needle/autograd.py#L763)), which is `reduce(add, [g])`
  ([autograd.py:855](../../../python/needle/autograd.py#L855)), and `reduce` returns `g` itself. So `.grad` and the list
  entry are one tensor, and dropping only one reference frees nothing for that node.
- **Lifetimes:** the lists are a local of `_compute_gradient_of_variables`
  ([autograd.py:768](../../../python/needle/autograd.py#L768)) and go away when backward returns; they add to the peak
  only. `.grad` stays on the node, so it also adds to the memory after backward (4943.7 vs 1092.7 MB), and the training
  loop keeps the graph until the next step (decision [05](../05-backward-release-scope.md), "Not in scope").

### Measurement: why the record takes only nodes that require grad (decision 02)

A node made by an op that does not require grad is temporary: `make_from_op` returns `tensor.detach()` in its place
([autograd.py:516-517](../../../python/needle/autograd.py#L516-L517)), and the detached leaf holds the same `NDArray`
([autograd.py:550](../../../python/needle/autograd.py#L550), [autograd.py:527](../../../python/needle/autograd.py#L527)).
Recording it frees nothing, backward never reaches it, and the record would keep it (and its buffer) alive until `fn`
returns.

Probe [probe_record_requires_grad.py](../../../agent-workspace/probe_record_requires_grad.py)
(log [probe_record_requires_grad.log](../../../agent-workspace/probe_record_requires_grad.log)): GPU 1, benchmark
transformer (activations size, float32, `checkpoint=True`); "all" also records the nodes that do not require grad.
Evaluation runs under `no_grad`, as `simple_ml` does ([simple_ml.py:165-168](../../../apps/simple_ml.py#L165-L168)).

| Mode | Record | Peak | Extra nodes recorded |
|---|---|---|---|
| eval (`no_grad`) | requires grad (`28fc334`) | 496.3 MB | 0 |
| eval (`no_grad`) | all | **1321.3 MB** | 560 |
| train | requires grad (`28fc334`) | 3014.2 MB | 0 |
| train | all | 3014.2 MB | 16 (`BroadcastTo` 8, `Reshape` 4, `Negate` 4) |

Loss 18.453640 in all runs. The 16 training nodes are ops on constants: `max_val` in the attention softmax
([nn_transformer.py:91-94](../../../python/needle/nn/nn_transformer.py#L91-L94)) and the causal mask
([nn_transformer.py:134](../../../python/needle/nn/nn_transformer.py#L134)).

Limit (not from the condition, from the detach in `make_from_op`): a detached leaf that an interior node uses as an
input stays in the graph ([autograd.py:390](../../../python/needle/autograd.py#L390)), and `checkpoint()` cannot
release it, because a leaf cannot be recomputed ([autograd.py:336](../../../python/needle/autograd.py#L336)).

## Commits

Full suite before the code commit: `904 passed in 36.05s`
([agent-workspace/pytest_before_commit.log](../../../agent-workspace/pytest_before_commit.log)).

| # | Decision | By | Reason |
|---|---|---|---|
| 1 | Two commits: the code first (`28fc334`: `autograd.py`, `__init__.py`, `nn_transformer.py`, `models.py`, `benchmark.py`, `tests/checkpoint/`), then the docs | proposed by Claude, accepted by the user | the code diff (+444 / −44) stays easy to review next to the docs (+1958); the docs cover all four steps (also `d011880`) and link to each other; the docs can name the code commit |
| 2 | The user's `.gitignore` change (`agent-workspace/`) goes into the docs commit | Claude | it concerns the agent folders, not the library |
| 3 | The logs and the update section of decision 05 refer to `28fc334`; the plan and decisions 01–11 keep `83ae951` | Claude | the links open the current files; log 03 was off by one or two lines after the docstring reflow and the user's comment |

The docs link to files in `agent-workspace/`. `.gitignore` excludes that folder, so these links work only in this
checkout.
