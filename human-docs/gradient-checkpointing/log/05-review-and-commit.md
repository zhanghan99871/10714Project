# Log 05: The review of `autograd.py` and the commits

Date: 2026-11-15. Branch: `gradient-checkpointing`. Status: done.
The full version is in [agent-docs/gradient-checkpointing/log/05-review-and-commit.md](../../../agent-docs/gradient-checkpointing/log/05-review-and-commit.md).

## The review

- You read the other files. Claude explained `autograd.py` to you.
- You changed the comment above the flag in `Value._init`. The commit keeps your comment.

## Findings

1. **A nested `checkpoint()` call gives an error.** Decision 04 says that the code raises no error of its own. But the inner call sets the record to `None` when it ends. Then the outer call fails with `TypeError: 'NoneType' object is not iterable`. We did not change the code.
2. **Backward must drop both `.grad` and the list of partial gradients.** On the GPU, with the benchmark transformer:
   - Release of the data only: peak 6266 MB.
   - Also drop `.grad`: peak 5962 MB.
   - Also drop the lists: peak 5850 MB.
   - Drop both: peak 3014 MB.

   For 500 of 540 interior nodes, `.grad` and the one entry in the list are the same tensor. Thus, one drop alone frees almost nothing. The lists go away when backward ends. `.grad` stays after backward: 4944 MB without the drop and 1093 MB with it.
3. **The record takes only nodes that require grad.** A node that does not require grad is temporary, because `make_from_op` puts a detached copy in its place. If the record also took these nodes, it would keep them alive until `fn` ends. During evaluation under `no_grad`, the peak would increase from 496 MB to 1321 MB. During training, the peak does not change.

## Decisions in this step

1. Two commits. First the code (`28fc334`), then the docs. Claude proposed this, and you accepted it.
2. Your `.gitignore` change goes into the docs commit (Claude).
3. The line numbers in the logs and in the update to decision 05 refer to `28fc334`. The plan and decisions 01 to 11 keep `83ae951` (Claude).

## Note

The docs link to files in `agent-workspace/`. `.gitignore` excludes that folder. Thus, these links work only in this copy of the repository.
