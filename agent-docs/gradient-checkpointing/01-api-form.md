# Decision 01 — API form

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md), step 1.

## Decision

Implement the function **`checkpoint(fn, *args)` first**. A module wrapper and `checkpoint_sequential` come **later**
(deferred, not rejected). The function form itself was already part of the user's plan ("create segments by
wrapping functions in a checkpoint() function"); this decision fixes the scope of the first implementation.

## Context

PyTorch offers four forms. All non-function forms are built on top of the function.

## Comparison with PyTorch

Reference: PyTorch 2.10.0 (env `vda`).

| PyTorch form | How it works | needle |
|---|---|---|
| `torch.utils.checkpoint.checkpoint(fn, *args)` (public) | Runs `fn` as one segment | **Now** |
| `torch.utils.checkpoint.checkpoint_sequential(functions, segments, input)` (public, `checkpoint.py` L520, L595-610) | Splits a `Sequential` into `segments` equal parts; checkpoints all but the last part ("the last chunk has to be non-volatile": backward needs it at once) | Later |
| `checkpoint_wrapper(module)` (private, `torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py` L195) | Returns a module whose every call runs `checkpoint()` | Later |
| Composable `checkpoint(module)` (private, `torch/distributed/_composable/checkpoint_activation.py` L38-46) | Registers pre-/post-forward hooks; model structure unchanged | Not possible: needle has no hooks; `Module.__call__` only calls `forward` ([nn_basic.py:73-74](../../python/needle/nn/nn_basic.py#L73-L74)) |

```python
# torch/utils/checkpoint.py L598-610 (checkpoint_sequential)
segment_size = len(functions) // segments
# the last chunk has to be non-volatile
end = -1
for start in range(0, segment_size * (segments - 1), segment_size):
    end = start + segment_size - 1
    input = checkpoint(run_function(start, end, functions), input, ...)
return run_function(end + 1, len(functions) - 1, functions)(input)
```

## Justification

1. The function is the core: the wrapper and `checkpoint_sequential` are thin layers on it (`checkpoint_sequential` is
   about 10 lines). Nothing in them is needed to prove the mechanism.
2. One entry point keeps the first change small and keeps the tests focused on the mechanism (decisions 02 and 05).
3. Adding the other forms later does not change the function's behaviour.

## Notes for the later forms

- A wrapper is cheap in needle: `Module.parameters()`, `train()` and `eval()` walk `__dict__`
  ([nn_basic.py:56-71](../../python/needle/nn/nn_basic.py#L56-L71)), so `self.module = module` exposes the wrapped
  module's parameters and training flag with no extra code. It would let a model constructor take a `checkpoint`
  option without a change to `forward` (e.g. wrap each `TransformerLayer`,
  [nn_transformer.py:346-357](../../python/needle/nn/nn_transformer.py#L346-L357)).
- With the function only, checkpointing the transformer layers means calling `checkpoint(layer, x)` in
  `Transformer.forward` or in the benchmark code (open decision: benchmark and integration).

## Location and public name (accepted 2026-11-14)

**Decision.** `checkpoint()` lives in [python/needle/autograd.py](../../python/needle/autograd.py), next to `no_grad`,
and is exported as **`ndl.checkpoint`** from [\_\_init\_\_.py](../../python/needle/__init__.py) (like
`from .autograd import no_grad, ...`, [\_\_init\_\_.py:4](../../python/needle/__init__.py#L4)).

**Options considered.**
- In `autograd.py`, like `no_grad` ([autograd.py:180-216](../../python/needle/autograd.py#L180-L216)) — chosen.
- In a new module, like `amp.py`: state in `autograd.py` (`_AutocastState`,
  [autograd.py:71-83](../../python/needle/autograd.py#L71-L83)), public `autocast` in `amp.py` changing
  `autograd._AUTOCAST` ([amp.py:91-113](../../python/needle/amp.py#L91-L113)).

**Comparison with PyTorch.** `torch.utils.checkpoint` is a module and `torch.utils.checkpoint.checkpoint` the function;
`torch` has no top-level `checkpoint`. needle has no `utils` package and exposes `no_grad` and `autocast` at the top level.

**Justification.**
1. The mechanism must be in `autograd.py` anyway: the record hook in `Tensor.make_from_op`
   ([autograd.py:493-504](../../python/needle/autograd.py#L493-L504), decision 02), the release in
   `_compute_gradient_of_variables` ([autograd.py:720-743](../../python/needle/autograd.py#L720-L743), decision 05), and
   the segment state next to `_GRAD_MODE` ([autograd.py:165-172](../../python/needle/autograd.py#L165-L172)).
2. The public function is small (about 60 lines with its docstring), and `no_grad` already lives there.
3. No new module, so no name clash. A module `checkpoint.py` whose package also exports a function `checkpoint` makes
   `import needle.checkpoint as m` return the function. Probe
   [agent-workspace/probe_name_clash.py](../../agent-workspace/probe_name_clash.py):
   ```
   import pkg.checkpoint as m -> function
   sys.modules["pkg.checkpoint"] -> module
   ```
