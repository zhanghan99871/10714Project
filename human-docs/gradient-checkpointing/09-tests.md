# Decision 09: Correctness tests

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/09-tests.md](../../agent-docs/gradient-checkpointing/09-tests.md).

## Decision

The tests go in a new file, `tests/checkpoint/test_checkpoint.py`. They run on the numpy, cpu and cuda devices.

Working rule: Claude can add, change or remove tests without asking. The list below is the start.

| No. | What the test checks |
|---|---|
| 1 | The gradients with and without `checkpoint()` are bit-identical, for an MLP, a `TransformerLayer` and an `LSTM`. |
| 2 | Test 1 also passes under float16 and bfloat16 autocast. |
| 3 | With `Dropout` and a fixed seed, the gradients with and without `checkpoint()` are identical. |
| 4 | With `BatchNorm`, the running statistics change one time per forward pass. |
| 5 | After forward, the buffers of interior nodes are freed, and the segment outputs keep their data. |
| 6 | After backward, the recomputed interior nodes are released again, and the outputs keep their data. |
| 7 | An op that counts its calls runs one time in forward, and at most one more time in backward. |
| 8 | In an `LSTM` segment, the `Split` and `MakeTensorTuple` nodes are released. |
| 9 | A nested return value such as `output, (h, c)` keeps all three tensors. |
| 10 | Under `no_grad`, `checkpoint()` gives the same result as `fn` and releases nothing. |
| 11 | An interior tensor that code reads after `opt.step()` has the new-weight value. |
| 12 | A second backward through the same graph gives the same gradients. |

The measurement tests (memory and recompute) need a memory counter. The next decision is about that counter.

## Update: as implemented (log 03)

- The file has 13 tests, which give 60 cases on numpy, cpu and cuda. All 60 pass.
- Test 6 also checks that backward drops the `.grad` of interior nodes.
- A new test checks decision 07: an explicit cast in a segment releases the segment input, and the gradients stay identical.

## Why the gradients are bit-identical

- The recompute runs the same ops on the same inputs.
- The graph does not change. Thus, backward adds the gradients in the same order.
- An existing test already shows that a recompute is bit-exact on all three devices.
