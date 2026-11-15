# Decision 01: The form of the API

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/01-api-form.md](../../agent-docs/gradient-checkpointing/01-api-form.md).

## Decision

We make the function `checkpoint(fn, *args)` first. We make a wrapper module and `checkpoint_sequential` later.

## Comparison with PyTorch

PyTorch has four forms:

| PyTorch form | How it works | Needle |
|---|---|---|
| `checkpoint(fn, *args)` | Runs `fn` as one segment. | Now |
| `checkpoint_sequential(functions, segments, input)` | Divides a `Sequential` into parts. It checkpoints each part except the last one, because backward needs the last part immediately. | Later |
| `checkpoint_wrapper(module)` (private) | Gives a module that runs `checkpoint()` at each call. | Later |
| Composable `checkpoint(module)` (private) | Adds forward hooks to the module. | Not possible, because needle has no hooks. |

## Justification

1. The function is the core. The other forms are thin layers on top of it.
2. One entry point keeps the first change small, and the tests can focus on the mechanism.
3. When we add the other forms later, the function does not change.

## Results

- To checkpoint the layers of the `Transformer` now, the code must call `checkpoint(layer, x)` in `Transformer.forward` or in the benchmark.
- A wrapper module is easy to add later. `parameters()`, `train()` and `eval()` find the wrapped module with no extra code.

## Location and public name

Decision: `checkpoint()` goes in `autograd.py`, next to `no_grad`. The package exports it as `ndl.checkpoint`.

- **Comparison with PyTorch:** PyTorch puts the function in the module `torch.utils.checkpoint`. Needle has no `utils` package, and it shows `no_grad` and `autocast` at the top level.
- **Justification:**
  1. The record hook, the release after backward and the segment state must be in `autograd.py` in any case.
  2. The public function is small, and `no_grad` is already in `autograd.py`.
  3. A new module named `checkpoint.py` would cause a name problem: `import needle.checkpoint as m` would give the function, not the module. A test shows this.
