# Decision 02: How `checkpoint()` finds the nodes of a segment

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/02-segment-membership.md](../../agent-docs/gradient-checkpointing/02-segment-membership.md).

## Decision

`checkpoint()` records each tensor that an op makes while the segment runs. At the end of the segment, it releases the recorded non-leaf tensors. It does not release the segment outputs.

## Options

- **Walk back from the outputs.** The walk goes through `inputs` and stops at the segment inputs and at leaves. But `fn` can use a non-leaf tensor that it does not get as an argument. Examples are a tensor from an outer scope, or a parameter cast that autocast made in an earlier segment. Then the walk goes outside the segment and releases earlier nodes. The gradients stay correct, but backward recomputes more than necessary.
- **Record the tensors that ops make (chosen).** This gives exactly the nodes that the segment made.

## Comparison with PyTorch

- **`use_reentrant=True`** does not track tensors in the segment. It runs `fn` under `no_grad`, so it records no graph. It keeps only the inputs.
- **`use_reentrant=False`** tracks each tensor that an op saves for backward. It puts an empty holder in place of the tensor. It finds the segment by time: a tensor belongs to the segment if an op saves it while the hook is active.
- **Our decision** also finds the segment by time. But it records the tensors that ops make, not the tensors that ops save. Needle has no separate saved tensor: each node keeps its output in `cached_data`, and any gradient can read it.

## Justification

1. The set of nodes is exact. `checkpoint()` never releases nodes from earlier code.
2. It uses the same principle as the recommended PyTorch mode.
3. It is cheap. It adds one check per op and needs no walk through the graph.

## Results

- If segment 1 makes an autocast cast of a parameter and segment 2 uses it, segment 1 releases it. Segment 2 recomputes it, and then it stays in memory.
- If `fn` keeps a tensor (for example, `self.cache = h`), `checkpoint()` releases it too. A later read recomputes it.

## Record storage

Decision: the record is a plain list with strong references. `checkpoint()` drops the list at the end of the segment.

- **Other option:** weak references. They free a tensor at once if nothing else uses it, but they need more code.
- **Test:** in `TransformerLayer` and `LSTM`, each tensor that the ops make is on a path to an output. The output graph keeps these tensors alive in any case. Thus, strong references keep no extra memory.
- **Comparison with PyTorch:** PyTorch uses weak references, because its record lives until backward. Our record lives only until the end of the segment.
- **Justification:** It is the simplest code, and it has no memory cost.
