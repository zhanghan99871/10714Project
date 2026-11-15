# Decision 08 — `no_grad`, no tensor that requires grad, and `LAZY_MODE`

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md).

## Decision

- **`no_grad`, and `fn` uses no tensor that requires grad** (neither the arguments nor the parameters): **no code**.
  `checkpoint()` records nothing, releases nothing, and only returns the result of `fn`.
- **`LAZY_MODE`: not supported.** One line in the `checkpoint()` docstring says so; no code.

## Why no code is needed for the first two cases

1. `_init` computes `requires_grad = _GRAD_MODE.enabled and any(x.requires_grad for x in inputs)`
   ([autograd.py:378](../../python/needle/autograd.py#L378)) → `False` in both cases.
2. `make_from_op` then returns `tensor.detach()`, a leaf with `op=None` and no inputs
   ([autograd.py:501-502](../../python/needle/autograd.py#L501-L502)).
3. The record hook (decision [06](06-tensortuple.md): in `Value._init`) records only nodes that come from an op and
   require grad, so nothing is recorded.

Probe [agent-workspace/probe_no_requires_grad.py](../../agent-workspace/probe_no_requires_grad.py):
```
W.requires_grad=False: h: requires_grad=False op=None   inputs=0 | y: requires_grad=False op=None   inputs=0
W.requires_grad=True : h: requires_grad=True  op=MatMul inputs=2 | y: requires_grad=True  op=ReLU   inputs=1
backward: backward() on a tensor that does not require grad: it was cr...
```
The second line is the normal training case: the input data does not require grad, but the parameters do, so the
nodes are graph nodes and `checkpoint()` works normally.

## `LAZY_MODE`

Ops do not compute when they are made, so at the end of the segment no node has data and the release does nothing
(decision 06). The flags would still make backward release nodes that it processes, and later reads would recompute
them. This path is untested. Only tests set `LAZY_MODE`
([test_cast_release.py:117](../../tests/fp16test/test_cast_release.py#L117), [test_no_grad.py:283](../../tests/unit/test_no_grad.py#L283),
[test_autocast.py:80](../../tests/fp16test/test_autocast.py#L80)); autocast ignores it ([amp.py:26](../../python/needle/amp.py#L26)).

## Comparison with PyTorch

- Non-reentrant: under `no_grad` it only runs `fn` (`torch/utils/checkpoint.py` L1602-1605, "When ambient grad_mode is False").
- Reentrant: needs an input that requires grad; if no output requires grad it raises "none of output has
  requires_grad=True, this checkpoint() is not necessary" (L322).
- needle behaves like the non-reentrant mode. PyTorch has no lazy mode.

## Justification

1. The current code already gives the correct behaviour for `no_grad` and for a `fn` without grad.
2. `LAZY_MODE` is used only by three tests; a docstring line is the simplest choice, as for nested segments (decision [04](04-nested-segments.md)).
