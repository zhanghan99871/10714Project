# Decision 05: Which nodes backward releases

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/05-backward-release-scope.md](../../agent-docs/gradient-checkpointing/05-backward-release-scope.md).

## Decision

After backward processes a node, it releases the node only if the node is an interior node of a checkpoint segment. Segment outputs and nodes outside segments keep their data.

## Options

- **Only the interior nodes of segments (chosen).** After backward, the memory is the same as after forward.
- **Every non-leaf node.** This gives memory back sooner. But it causes wrong values, and no error tells you:
  - A recompute uses the current data of the leaves.
  - The optimizer gives the parameters new data.
  - The training loop reads `output.numpy()` after the optimizer step ([simple_ml.py:233-235](../../apps/simple_ml.py#L233-L235)).
  - A test shows the effect: the output is `[3. 3.]` before the step and `[2.4 2.4]` when the loop reads it after the step.

## Comparison with PyTorch

- **Normal backward:** PyTorch frees the graph after backward by default (`retain_graph=False`). But it frees only the tensors that the graph saved. A tensor that your code holds keeps its value. Needle has only one buffer per node, so a release also removes your value.
- **Checkpoint:** PyTorch clears each recomputed tensor when backward uses it ("Rule 4" in `torch/utils/checkpoint.py`). Our decision applies the same rule.

## Justification

1. Normal code gets no wrong values. A read of the output after the optimizer step stays correct.
2. It agrees with the checkpoint rule in PyTorch.
3. Code without `checkpoint()` works as before.
4. During backward, only one segment at a time has recomputed data.

## How backward finds the interior nodes

Decision: each node has a flag. When `checkpoint()` releases a node at the end of the segment, it sets the flag. After backward processes a node, it releases the node if the flag is set.

- **Other option:** a global set of interior nodes. This set needs cleanup.
- **Comparison with PyTorch:** PyTorch also keeps the mark in the graph. It puts a holder in place of each saved tensor.
- **Justification:** It needs one attribute and one check, and no cleanup.

## Update: backward also drops the gradients of interior nodes (Claude, during the implementation)

- When backward releases an interior node, it also removes the node's `.grad` and its partial gradients. Nodes outside segments keep their `.grad`.
- **Why:** backward keeps the gradient of each node. With checkpoints, these gradients became the largest part of the peak memory. In a test on the CPU, the peak was 194 MB without checkpoints, 118 MB with checkpoints, and 51 MB when backward also dropped the interior gradients. The gradients stayed bit-identical. On the GPU, with the benchmark transformer, the peak was 6266 MB without the drop and 3014 MB with it (log 05).
- **Comparison with PyTorch:** PyTorch keeps `.grad` only on leaves. Needle keeps it on each node, and an existing test reads it. Thus, the change applies only to interior nodes.

## Limits

- If code reads an interior tensor of a segment after the optimizer step, it gets a value made with the new weights. No error tells you.
  - A test shows this. An interior node gives `[3. 3. 3.]` in forward and `[2.4 2.4 2.4]` after the step. The segment output keeps its forward value.
  - The same problem exists today for an input that `ops.cast` releases.
  - Only code that keeps an interior tensor has this problem. No needle module does this today.
  - The docstring of `checkpoint()` tells the user about this limit. PyTorch also puts its recompute warning in the docstring of `checkpoint()`.
- A second backward through the same graph works. It recomputes the interior nodes again.
- The graph of the previous step stays in memory until the code stops holding `output` and `loss`. This problem is not part of checkpointing.
