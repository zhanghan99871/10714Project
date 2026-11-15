# Decision 05 — Which nodes backward releases after processing them

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md), step 4.

## Decision

After `_compute_gradient_of_variables` has processed a node, it releases the node's `cached_data` **only if the node
is an interior node of a checkpoint segment**, i.e. a node that `checkpoint()` released at the end of its segment
(decision [02](02-segment-membership.md)). Segment outputs and all nodes outside segments keep their data, as today.

## Context

The probe in [00-plan.md](00-plan.md) §2.4 shows that recomputed nodes keep their data after backward. Step 4 releases
them again. Releasing is safe at that point because backward visits nodes in reverse topological order
([python/needle/autograd.py:716-743](../../python/needle/autograd.py#L716-L743)): every consumer of the node has been
processed, so no later gradient in this pass reads it.

## Options considered

### A. Only the interior nodes of segments (chosen)
After backward, memory returns to the state after forward.

### B. Every non-leaf node
Also releases segment outputs and all nodes outside segments.

Benefit: memory returns sooner. Today needle keeps every activation alive while something holds the graph, and a
training loop holds the previous step's `output` and `loss` during the next forward. Probe
[agent-workspace/probe_needle_graph_lifetime.py](../../agent-workspace/probe_needle_graph_lifetime.py):

```
step 0: hidden buffer alive after backward + step: True
  at start of next forward, previous hidden buffer alive: True
```

Problem: **wrong values without an error.** A recompute uses the *current* data of the leaves. Optimizers replace
parameter data (`param.data -= self.lr * ut`, [optim.py:38](../../python/needle/optim.py#L38); the setter assigns a new
buffer, [autograd.py:523-531](../../python/needle/autograd.py#L523-L531)). The training loop reads the model output
after the step ([apps/simple_ml.py:233-235](../../apps/simple_ml.py#L233-L235)):

```python
if opt is not None:
    _optimizer_step(loss, opt, scaler)
logits = output.numpy()
```

Under B, `output` would be recomputed with the new weights. Probe
[agent-workspace/probe_recompute_after_step.py](../../agent-workspace/probe_recompute_after_step.py):

```
out before step: [3. 3.]
out read after step: [2.4 2.4]
```

The reported accuracy would be wrong.

## Comparison with PyTorch

Reference: PyTorch 2.10.0 (env `vda`).

- **General backward (≈ option B, but without the problem).** With `retain_graph=False` (default) PyTorch frees the
  graph after backward (`torch/autograd/__init__.py` L304-307: *"If ``False``, the graph used to compute the grad will be
  freed."*). It frees only the references that the graph holds to its **saved tensors**; a tensor that user code holds
  keeps its value. Probe [agent-workspace/probe_torch_free_after_backward.py](../../agent-workspace/probe_torch_free_after_backward.py):
  ```
  saved tensor after backward: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)
  user-held y still readable: True
  ```
  needle has one buffer per node: `cached_data` is both the saved tensor and the user's value. Releasing it removes the
  user's value too, which causes the problem above. The needle equivalent of PyTorch's behaviour would be to drop the
  graph edges (`inputs`) after backward and keep the data — a separate change, not part of checkpointing.
- **Checkpoint-specific (= option A).** PyTorch clears each recomputed tensor as soon as backward unpacks it, also with
  `retain_graph=True` (`torch/utils/checkpoint.py` L659-662, "Rule 4": *"Recomputed tensors are considered specific to
  particular invocations of backward and are always cleared immediately as they are unpacked"*; code: L1196
  `holder.handles[gid] = None`). Option A applies the same rule to needle's recomputed nodes.

## Justification

1. **No wrong values in normal code.** Segment outputs and nodes outside segments keep their data, so reads such as
   `output.numpy()` after the optimizer step stay correct.
2. **Matches PyTorch checkpoint semantics** (Rule 4): recomputed activations live only during the backward pass that
   needed them.
3. **No behaviour change without checkpoints.** Code that does not call `checkpoint()` behaves exactly as before; the
   existing 828 tests are unaffected by this rule.
4. **Peak memory as intended.** During backward, at most one segment's recomputed activations are alive at a time
   (plus the checkpoints).

## Consequences and limits

- **Read-after-step limit.** Reading an **interior** tensor of a segment after `opt.step()` recomputes it with the
  new weights. Probe [agent-workspace/probe_read_after_step.py](../../agent-workspace/probe_read_after_step.py):
  ```
          h after backward, before decision-02 release: cached = True
  case 1: h in forward: [3. 3. 3.]  h read after step: [2.4 2.4 2.4]
          out (kept) after step: [9.]
  case 2: y in forward: [2.4 2.4 2.4]  y read after step: [1.8000001 1.8000001 1.8000001]
  ```
  Case 1: interior node → new value; segment output → forward value. Case 2: the same limit exists today for an input
  released by `ops.cast` ([ops_mathematic.py:64-72](../../python/needle/ops/ops_mathematic.py#L64-L72)); the `ops.Cast`
  docstring does not mention it ([ops_mathematic.py:34-37](../../python/needle/ops/ops_mathematic.py#L34-L37)).
  Only code that keeps an interior tensor is affected; no needle module does (`AttentionLayer` discards `prob`,
  [nn_transformer.py:248](../../python/needle/nn/nn_transformer.py#L248)).
  **Documented in the `checkpoint()` docstring** (user, 2026-11-14), together with the dict limit of decision
  [03](03-segment-outputs.md). PyTorch has no such limit (a held tensor keeps its value), and puts its own recompute
  warning in the same place (`torch/utils/checkpoint.py` L376-381).
- A second backward through the same graph works; it recomputes the interior nodes again.
- With `create_graph=True`, the gradient graph references forward nodes; released interior nodes are recomputed on demand.

## How backward finds interior nodes (accepted 2026-11-14)

**Decision.** A **flag on each node**. When `checkpoint()` releases a node at the end of its segment, it sets the flag;
`Value._init` sets it to `False` for every other node. After `_compute_gradient_of_variables` has processed a node
([autograd.py:720-743](../../python/needle/autograd.py#L720-L743)), it releases the node if the flag is set.

**Options considered.** A flag on the node (chosen), or a global set of interior nodes (needs cleanup, and either keeps
the nodes alive or needs weak references).

**Comparison with PyTorch.** PyTorch keeps the mark in the graph: the pack hook puts a `_Holder` in place of each saved
tensor (`torch/utils/checkpoint.py` L1144-1152), and the unpack hook clears it (L1196).

**Justification.** One attribute and one check; no cleanup; the flag goes away with the node.

## Update 2026-11-14: interior adjoints are dropped too (Claude, during the implementation)

**Decision.** When backward releases an interior node, it also sets `each.grad = None` and deletes the node's entry in
`node_to_output_grads_list` ([autograd.py:803-808](../../python/needle/autograd.py#L803-L808) in `28fc334`). Nodes outside segments
keep their `.grad`, as before.

**Why.** Backward stores the adjoint of every node in `.grad` ([autograd.py:782](../../python/needle/autograd.py#L782) in `28fc334`)
and keeps every partial adjoint in `node_to_output_grads_list` until it returns. With checkpoints these adjoints became
the largest part of the peak. Probe [agent-workspace/probe_checkpoint_memory.py](../../agent-workspace/probe_checkpoint_memory.py)
(cpu, transformer LM, 4 layers, batch 8, seq 64, one step; `.grad` buffers counted once):
```
release only:
no checkpoint: after forward   93.4 MB | peak  194.1 MB | after backward  167.2 MB | of which non-leaf .grad   71.8 MB
checkpoint   : after forward   17.2 MB | peak  117.9 MB | after backward   91.0 MB | of which non-leaf .grad   71.8 MB
release + drop interior adjoints:
checkpoint   : after forward   17.2 MB | peak   50.5 MB | after backward   20.2 MB | of which non-leaf .grad    0.9 MB
```
Gradients stayed bit-identical. On the GPU, with the benchmark transformer, the peak is 6266 MB with the release only and 3014 MB with both drops (log [05](log/05-review-and-commit.md)). Nothing reads an interior adjoint after its node is processed (each gradient op uses
only its own node's adjoint), and interior nodes are segment-internal by decision [02](02-segment-membership.md).
Tested by `test_interior_released_after_backward`.

**Comparison with PyTorch.** PyTorch keeps `.grad` only on leaves; a non-leaf `.grad` needs `retain_grad()`. needle
keeps it on every node; the existing test `test_gradients_are_not_released` reads non-leaf `.grad`
([test_cast_release.py:223-224](../../tests/fp16test/test_cast_release.py#L223-L224)), so the change is limited to
interior nodes.

## Not in scope

- The previous step's graph (checkpoints and nodes outside segments) stays alive until user code drops `output` and
  `loss`. Fixing this is the "drop graph edges after backward" change above, independent of checkpointing.
