# Decision 02 — How `checkpoint()` finds the nodes of a segment

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md), step 2.

## Decision

`checkpoint()` **records every tensor that an op makes while the segment runs** (a hook in `Value._init`, see decision [06](06-tensortuple.md);
active only between the start and the end of the segment). At the end of the segment it releases the recorded
non-leaf tensors, except the segment outputs.

## Context

Step 2 of the plan releases "the activations of the segment". The code needs an exact set of nodes for this.

## Options considered

### A. Walk back through `inputs` from the outputs
Stop at the segment inputs and at leaves; release every non-leaf node on the way.

Problem: `fn` can use a non-leaf tensor that is not one of its arguments. The walk then leaves the segment and
releases nodes of earlier parts of the graph. Examples:
- a tensor from an enclosing scope (closure, module attribute);
- a parameter cast that autocast made in an earlier segment and keeps in its cache
  ([python/needle/autograd.py:113-122](../../python/needle/autograd.py#L113-L122)):
  ```python
  if _AUTOCAST.cache_enabled and x.is_leaf() and x.requires_grad:
      key = (id(x), dtype, _GRAD_MODE.enabled)
      hit = _AUTOCAST.cache.get(key)
      if hit is not None and hit[0] is x and hit[1] is x.cached_data:
          return hit[2]
  ```

Gradients stay correct (released nodes are recomputed), but backward recomputes more than necessary, and the
amount depends on how `fn` gets its tensors.

### B. Record the tensors that ops make during the segment (chosen)
Gives exactly the nodes that the segment created, independent of where `fn` gets its inputs.

## Comparison with PyTorch

Reference: PyTorch 2.10.0, `torch/utils/checkpoint.py` (env `vda`).

| | PyTorch `use_reentrant=True` | PyTorch `use_reentrant=False` | needle (this decision) |
|---|---|---|---|
| Forward | `fn` under `torch.no_grad()` (L267-268): no graph inside, only the inputs are saved | `fn` with grad on, inside `_checkpoint_hook` (`saved_tensors_hooks`, L1607) | `fn` with grad on; `make_from_op` records each new tensor |
| What is tracked | nothing inside the segment | each tensor that an op **saves for backward**: pack hook stores an empty `_Holder` (L1144-1152) | each tensor that an op **makes** |
| How the segment is delimited | the segment is one `autograd.Function` node | by time: whatever is saved while the hook is active | by time: whatever is made while recording is active |
| Recompute | run `fn` again in backward (L307-325) | run `fn` again on first unpack (L1164-1173, L1591); a second hook refills holders by a counter (L1081-1120) | `op.compute()` on recorded nodes via `realize_cached_data()` |

```python
# torch/utils/checkpoint.py L1144-1152 (_checkpoint_hook.pack_hook)
def pack_hook(x):
    holder = _Holder()
    frame.weak_holders.append(weakref.ref(holder))
    ...
    return holder
```

Why needle records *made* tensors, not *saved* tensors: needle has no separate "saved tensor" object. Every node
keeps its output in `cached_data`, and any gradient may read `node` or `node.inputs`
([ops_mathematic.py:117-119](../../python/needle/ops/ops_mathematic.py#L117-L119),
[ops_mathematic.py:531-534](../../python/needle/ops/ops_mathematic.py#L531-L534)). So the buffer that backward reads is the
node's own output, and the unit to track is the node.

## Justification

1. **Exact set.** Only nodes created by the segment are released; nodes from earlier code are never touched (option
   A's failure cases do not apply).
2. **Same principle as the recommended PyTorch mode.** PyTorch non-reentrant delimits the segment by the time window in
   which its hook is active, not by walking the graph.
3. **Cheap.** One check in `make_from_op` per op while a segment is active; no graph traversal at the end of the segment.

## Consequences

- Autocast parameter casts created in segment 1 and reused in segment 2 are recorded by segment 1, released at its end,
  recomputed (a cheap cast) when segment 2 uses them, and then stay cached (segment 2 did not make them).
- Tensors that the segment makes but that `fn` stores somewhere (e.g. `self.cache = h`) are released too; reading
  them later recomputes them (see the recompute-after-step hazard in [05-backward-release-scope.md](05-backward-release-scope.md)).
- Detached results (`requires_grad=False`, returned by `make_from_op` as constants,
  [autograd.py:501-502](../../python/needle/autograd.py#L501-L502)) are leaves and are not released.

## Record storage (accepted 2026-11-14)

**Decision.** The record is a **plain list with strong references**; `checkpoint()` drops it at the end of the segment.

**Options considered.** Strong references (chosen) keep a tensor alive until the segment ends even if nothing else
uses it; weak references (`weakref.ref`) free such a tensor at once but need more code.

**Evidence.** The difference matters only for tensors that the output graph does not hold (each node holds its
inputs). Probe [agent-workspace/probe_dead_temporaries.py](../../agent-workspace/probe_dead_temporaries.py):
```
TransformerLayer: made 136, not reachable from outputs 0 (0 bytes) []
LSTM: made 613, not reachable from outputs 0 (0 bytes) []
```

**Comparison with PyTorch.** The frame keeps `weakref.ref(holder)` for each saved tensor
(`torch/utils/checkpoint.py` L1146-1147). The frame lives until backward, so strong references would keep holders alive
after the graph frees them. Our record lives only until the end of the segment.

**Justification.** Simplest code; no measured memory cost.

