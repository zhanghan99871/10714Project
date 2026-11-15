# Gradient checkpointing: the plan

Date: 2026-11-14. Status: steps 1 to 4 are done and committed. Steps 1 and 2 are in commit `d011880`. Steps 3 and 4 are in commit `28fc334`. Log 05 records the review and the commits.
The full version with all references is in [agent-docs/gradient-checkpointing/00-plan.md](../../agent-docs/gradient-checkpointing/00-plan.md).

## The plan

1. `checkpoint(fn, *args)` runs `fn` as one segment.
2. At the end of the segment, `checkpoint()` releases the data of the nodes in the segment. It keeps the segment inputs and outputs.
3. During backward, `realize_cached_data()` recomputes a released node when a gradient needs it.
4. After backward processes a node, backward releases that node again.

## Why the plan works with the current code

- `realize_cached_data()` recomputes a released node from its inputs ([autograd.py:315-333](../../python/needle/autograd.py#L315-L333)).
- `release_cached_data()` removes the data and keeps the shape, dtype and device ([autograd.py:335-346](../../python/needle/autograd.py#L335-L346)).
- Gradients read `node` and `node.inputs`. Each read makes a new op, and that op calls `realize_cached_data()`. Thus, step 3 needs no new code.
- Backward processes the nodes in reverse topological order ([autograd.py:716-743](../../python/needle/autograd.py#L716-L743)). When backward processes a node, it has already processed all the users of that node. Thus, step 4 is safe.
- A test shows that the gradients stay bit-identical after a release. Backward recomputes only the nodes that a gradient reads.

## Comparison with PyTorch

- **PyTorch runs `fn` again during backward.** This is true for `use_reentrant=True` and also for `use_reentrant=False`. A test shows two calls of `fn`. The second call is inside backward.
- **Thus, PyTorch must make the second run the same as the first.** It saves and restores the random state and the autocast state. It does not stop other side effects. For example, BatchNorm updates its running statistics two times (`num_batches_tracked` goes from 1 to 2 during backward).
- **Our plan does not run `fn` again.** It runs `op.compute()` on the recorded nodes. Thus:
  - Dropout keeps its mask as a leaf. The recompute uses the same mask.
  - BatchNorm updates its statistics outside the graph. The recompute does not update them again.
  - The autocast `Cast` nodes are in the graph. The recompute uses the same dtypes.
- **Cost:** leaf constants in the segment stay in memory. For example, a Dropout mask is as large as its activation. PyTorch makes the mask again from the saved random state.

## Decisions

| No. | Question | Decision |
|-----|----------|----------|
| [01](01-api-form.md) | What is the form of the API (Application Programming Interface)? | `checkpoint(fn, *args)` first. A wrapper module and `checkpoint_sequential` later. The function goes in `autograd.py`, as `ndl.checkpoint`. |
| [02](02-segment-membership.md) | How does `checkpoint()` find the nodes of a segment? | It records each tensor that an op makes during the segment, in a plain list. It drops the list at the end of the segment. |
| [03](03-segment-outputs.md) | How does `checkpoint()` find segment outputs? | Each tensor in nested tuples and lists. Dicts are not supported yet, and the docstring tells this. |
| [04](04-nested-segments.md) | Does needle support nested segments? | No. The docstring tells this. `checkpoint()` does not raise an error. |
| [05](05-backward-release-scope.md) | Which nodes does backward release? | Only the interior nodes of checkpoint segments. Each interior node has a flag. Backward also drops the gradients of interior nodes. The docstring of `checkpoint()` tells about the problem of a read after the optimizer step. |
| [06](06-tensortuple.md) | What does `checkpoint()` do with `TensorTuple` nodes? | It releases them as `Tensor` nodes. We remove the guard in `release_cached_data()`. |
| [07](07-cast-of-segment-inputs.md) | Does `checkpoint()` protect its inputs from a cast that releases them? | Not in the first version. Tests measure the effect first. |
| [08](08-special-modes.md) | What does `checkpoint()` do under `no_grad`, without grad, and in `LAZY_MODE`? | No code for `no_grad` and without grad. `LAZY_MODE` is not supported, and the docstring tells this. |
| [09](09-tests.md) | Which correctness tests do we write? | 12 tests in `tests/checkpoint/test_checkpoint.py`, on numpy, cpu and cuda. Claude can change the tests without asking. |
| [10](10-memory-measurement.md) | How do we measure memory? | A counter in Python, in `NDArray.make`, for each device. Three functions with the PyTorch names. |
| [11](11-benchmark.md) | What do we benchmark? | AMP first, on the Transformer and LSTM language models. Then checkpointing, on the Transformer language model, on GPU 1. |

## Known problems

- `release_cached_data()` fails on a `TensorTuple`. `ops.split` makes `TensorTuple` nodes, and the RNN (recurrent neural network) and LSTM (long short-term memory) modules use `ops.split`. Decision 06 removes this problem.
- A released node that code reads after the optimizer step gets the new weights. Decision 05 limits this problem to interior nodes.
- An explicit `ops.cast()` in a segment releases its input. A cast that autocast adds does the same under `autocast(release_inputs=True)`. Decision 07: no protection in the first version. If that input is a segment input, the recompute must go back into the earlier graph. The result is correct, but it takes more time.

## Open questions

None. All the design decisions have an answer.

## Working rules

- We work in the branch `gradient-checkpointing`. `main` does not change.
- Claude starts a step of work only when you say "start".
- The line numbers in the plan and in decisions 01 to 11 refer to commit `83ae951`. The later changes moved some lines in `ndarray.py` and `autograd.py`. The line numbers in the logs and in the update to decision 05 refer to commit `28fc334`.
- The folder [log/](log/) has one file for each step of work. Each file also records the decisions in that step.
- Claude can add, change or remove tests without asking.
- Claude can add or change benchmarks for VRAM and speed without asking.

## Order of work

1. The memory counter (decision 10) and its tests. The AMP benchmark needs the counter. Done and committed (log 01).
2. The AMP benchmark: write it, run it, and record the results. Done and committed (log 02).
3. `checkpoint()` and its tests. Done and committed (log 03).
4. The checkpoint benchmark. Done and committed (log 04).

Removed, because they have an answer: the checkpoint benchmark (decision 11), the method to measure memory (decision 10), the correctness tests (decision 09), `no_grad`, no tensor that requires grad, and `LAZY_MODE` (decision 08), casts that release a segment input (decision 07), what to do with `TensorTuple` nodes (decision 06), nested segments (decision 04), how backward finds the interior nodes (decision 05), strong or weak references (decision 02), the form of the API, the location of the code and its public name (decision 01), the outputs of `fn` (decision 03), and where to document the problem of a read after the optimizer step (decision 05).

## Environment

- Conda environment `needle`: Python 3.11, CMake 3.31, pybind11, numpy, ml_dtypes, pytest.
- Build command: `PATH=/home/xiachu/.conda/envs/needle/bin:$PATH make`.
- Baseline: 828 tests pass.
