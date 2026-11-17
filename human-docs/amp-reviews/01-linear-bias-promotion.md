# AMP review 01: `Linear` keeps the transformer in float32

Date: 2026-11-15. Branch: `gradient-checkpointing`, commit `33bdf5f`. The line numbers refer to this commit. The AMP code is the same as on `main` (`83ae951`), but some lines in `autograd.py` moved because of the checkpoint code.

## Summary

- Under AMP, the transformer uses almost the same memory as in float32. After forward, AMP uses 1.9% less memory. At the peak of backward, AMP uses 0.7% more memory.
- AMP is still faster: 1.38 times with float16 and 1.39 times with bfloat16 (checkpoint benchmark, log 04).
- **The main cause is `nn.Linear`.** It adds the bias in a separate op after the matrix multiplication. That op changes the float16 result back to float32. Thus, almost all activations stay in float32.
- Two more causes increase the memory:
  - Each cast keeps its input and makes a copy.
  - Each gradient has the dtype of its node, and the cast nodes get gradients too.

## How autocast selects a dtype

Each op has an autocast policy ([amp.py:11-19](../../python/needle/amp.py#L11-L19)):

| Policy | Ops | What autocast does |
|---|---|---|
| "lower" | `MatMul` ([ops_mathematic.py:396](../../python/needle/ops/ops_mathematic.py#L396)), `Conv` | It changes the inputs to float16 (or bfloat16). |
| "float32" | `Exp` ([ops_mathematic.py:479](../../python/needle/ops/ops_mathematic.py#L479)), `Log`, `Summation`, `EWisePow`, `PowerScalar`, `LogSoftmax`, `LogSumExp` | It changes the inputs to float32. |
| "promote" | all other ops (the default, [autograd.py:249](../../python/needle/autograd.py#L249)) | If the inputs have different dtypes, it changes all inputs to the widest dtype. If they have the same dtype, it changes nothing. |

The docstring of `amp.py` gives the problem as an example of "promote" ([amp.py:17-19](../../python/needle/amp.py#L17-L19)): "a float16 matmul result plus a float32 bias gives float32".

## The problem: `Linear` gives a float32 result

In needle, `Linear` is two ops ([nn_basic.py:96-106](../../python/needle/nn/nn_basic.py#L96-L106)):

```python
        res = X @ self.weight
        if self.need_bias: 
            bias_shape = (1,) * (len(res.shape) - 1) + (self.out_features,)
            bias = self.bias.reshape(bias_shape)
            bias = bias.broadcast_to(res.shape)
            return res + bias 
```

1. The matrix multiplication has the "lower" policy. Autocast changes `X` and the weight to float16, and the result is float16.
2. The bias is a float32 parameter. Autocast does not change it, because it is not an input of the matrix multiplication.
3. The add has the "promote" policy. It gets a float16 input and a float32 input. Thus, autocast changes the result of the multiplication back to float32.
4. The output of `Linear` is float32.

A test on the CPU shows this:

```
matmul output: float16 | Linear output: float32 | bias: float32
Linear(bias=False) output: float16
```

## Comparison with PyTorch

PyTorch uses the same promotion rule. The difference is where the bias goes. In PyTorch, `nn.Linear` is one op, and the bias is an input of that op. Before a float16 op, autocast changes all inputs of the op to float16. Thus, PyTorch changes the bias to float16 too.

We traced the ops that PyTorch 2.10.0 runs under `torch.autocast("cuda", dtype=torch.float16)`:

| Code | What PyTorch does | Result |
|---|---|---|
| `nn.Linear(x)` | It changes `x`, the weight and the bias to float16. Then one op (`addmm`) calculates `bias + x @ weight.T` in float16. | float16 |
| `x @ weight.T + bias` (two ops, as in needle) | It changes `x` and the weight to float16. The multiplication gives float16. The add gets float16 and float32. | float32 |
| `relu(nn.Linear(x))` | The ReLU runs in the dtype of its input. | float16 |

Thus, in PyTorch, the ops after `Linear` stay in float16. In needle, they run in float32.

## The effect in the transformer

- **The feed-forward part:** `linear1` and `linear2` have a bias ([nn_transformer.py:284-285](../../python/needle/nn/nn_transformer.py#L284-L285)). Their outputs are float32. Thus, the ReLU, the Dropout and the residual add after them are float32 too.
- **The attention part:** the projections have no bias ([nn_transformer.py:191-207](../../python/needle/nn/nn_transformer.py#L191-L207)), so their outputs are float16. But the softmax uses `Exp` ([nn_transformer.py:94](../../python/needle/nn/nn_transformer.py#L94)), which has the "float32" policy. So the attention scores are float32 from the `Exp` on. These are the largest tensors in the layer (batch × heads × sequence × sequence).
- **The residual add after attention:** it adds the float32 input of the layer ([nn_transformer.py:304](../../python/needle/nn/nn_transformer.py#L304)), so its result is float32.
- **LayerNorm:** it uses sums and powers, which have the "float32" policy.

## Two more causes

### Each cast keeps its input

- By default, a cast that autocast adds does not release its input (`self.release_inputs = False`, [autograd.py:78](../../python/needle/autograd.py#L78)). The docstring gives the reason ([amp.py:75-78](../../python/needle/amp.py#L75-L78)): an input of a float16 matrix multiplication is often used again, for example by a residual add.
- Needle keeps the data of all graph nodes until backward.
- Thus, each matrix multiplication keeps four tensors:
  - its float32 input
  - a float16 copy of that input
  - its float16 result
  - a float32 copy of that result, when a "promote" op follows

### Each gradient has the dtype of its node

- Backward makes sure that the gradient of a node has the dtype of that node ([autograd.py:789-798](../../python/needle/autograd.py#L789-L798)). So the float32 activations get float32 gradients.
- Each cast is a node in the graph, so backward also gives it a gradient.

## Measurements

Setup: the transformer of the benchmark at the "activations" size. It has 4 layers, embedding 256, hidden size 1024, batch 32, sequence length 256 and vocabulary 1000. We measured one training step without the optimizer, on an RTX A4000, with the memory counter of `needle.backend_ndarray`.

### The graph after forward

| | float32 | float16 |
|---|---|---|
| Memory after forward | 5384.4 MB | 5283.9 MB (−1.9%) |
| float32 buffers | 5376.4 MB | 4064.2 MB |
| float16 buffers | 0 | 1211.7 MB |
| Results of `MatMul` | 623.2 MB | 311.6 MB |
| Casts float16 → float32 of activations | 0 | 575.2 MB |
| Casts float32 → float16 of activations | 0 | 260.0 MB |

Without the casts, the float16 graph holds 936 MB less than the float32 graph. The casts add 835 MB, so the difference is only 100 MB.

### Forward and backward

| | float32 | float16 | float16 + `release_inputs=True` |
|---|---|---|---|
| Peak of forward | 5478.1 MB | 5377.5 MB | 4617.9 MB (−16%) |
| Memory after forward | 5384.4 MB | 5283.9 MB | 4524.2 MB (−16%) |
| **Peak of backward** | **10 510.9 MB** | **10 585.8 MB (+0.7%)** | **9826.1 MB (−6.5%)** |
| Memory after backward | 9330.7 MB | 9433.7 MB | 8674.1 MB |

The loss was 18.453640 in float32 and 18.453600 in float16. The full benchmark step includes the optimizer, and its peaks agree with these values: 10 543 MB in float32, 10 618 MB in float16 and in bfloat16, and 9858 MB in float16 with `release_inputs=True`.

### What holds the memory at the end of backward

Needle frees nothing during backward, so the peak of backward is close to the memory at its end.

| | float32 | float16 | float16 + `release_inputs=True` |
|---|---|---|---|
| Activations | 5376.4 MB float32 | 4064.1 MB float32, 1211.7 MB float16 | 3544.1 MB float32, 972.1 MB float16 |
| Gradients of activations | 3930.3 MB float32 | 3330.3 MB float32, 803.6 MB float16 | 3330.3 MB float32, 803.6 MB float16 |
| Lists of partial gradients | 1149.0 MB float32 | 1021.8 MB float32, 115.6 MB float16 | 1021.8 MB float32, 115.6 MB float16 |
| Gradients of parameters | 16.0 MB | 16.0 MB | 16.0 MB |

Under float16, the gradients of activations take 4134 MB. This is more than the 3930 MB in float32, because the cast nodes get gradients too.

## Related findings

1. **`release_inputs=True` frees the originals of the copies.** It frees 760 MB of activations: 520 MB float32 and 240 MB float16. Only 8 released tensors (32 MB) were computed again, all in forward. It does not change the gradients, so it lowers the peak of backward by only 6.5%.
2. **The casts in backward release nothing.** There are three reasons:
   - Autocast is off in backward ([autograd.py:743](../../python/needle/autograd.py#L743)).
   - Each cast in backward passes `release_input=False` ([ops_mathematic.py:59-61](../../python/needle/ops/ops_mathematic.py#L59-L61), [autograd.py:758](../../python/needle/autograd.py#L758), [autograd.py:762](../../python/needle/autograd.py#L762), [autograd.py:798](../../python/needle/autograd.py#L798)).
   - In backward, each gradient is a detached leaf, and a leaf cannot be computed again. So the release function skips it ([ops_mathematic.py:64-72](../../python/needle/ops/ops_mathematic.py#L64-L72)).

   If backward dropped the gradients of the cast nodes after it processed them, the peak of backward would fall by only 524 MB (5%). The cast nodes hold 811.4 MB of gradients, but other nodes also hold 287.4 MB of these buffers. For example, an add gives the same gradient tensor to both inputs ([ops_mathematic.py:90-91](../../python/needle/ops/ops_mathematic.py#L90-L91)).
3. **Dropping the gradients of all non-leaf nodes does not fix AMP.** This drop runs after backward processes each node. It is the same step that the checkpoint code does for interior nodes ([autograd.py:803-808](../../python/needle/autograd.py#L803-L808)), and PyTorch also keeps no `.grad` on non-leaf tensors. The parameter gradients stayed bit-identical.

   | Peak of backward | Drop nothing | Drop the gradients of all non-leaf nodes |
   |---|---|---|
   | float32 | 10 510.9 MB | 6436.1 MB |
   | float16 | 10 585.8 MB | 6215.6 MB |
   | float16 + `release_inputs=True` | 9826.1 MB | 5456.0 MB |

   The drop lowers the peak by about 40% for all dtypes. But float16 still saves only 3.4% compared with float32 (6216 MB compared with 6436 MB). The cause is again `Linear`.

## Not in this review

At the "weights" size of the benchmark, AMP uses 24% more memory, and float16 is slower than bfloat16. These problems have other causes: the half precision copies of the weights and their gradients, and the time of `GradScaler.unscale_`. Log 02 of gradient checkpointing has the details.

## Evidence

The test scripts are in `agent-workspace/`, which is not in git:

| Script | What it measures |
|---|---|
| `probe_linear_dtype.py` | the output dtype of `Linear` under autocast |
| `probe_torch_linear_trace.py` | the ops that PyTorch autocast runs for `Linear` |
| `probe_needle_linear_trace.py` | the graph nodes that needle autocast makes for `Linear` |
| `probe_amp_activation_dtypes.py` | the graph after forward, by dtype and by op |
| `probe_amp_step_memory.py` | forward and backward, `release_inputs`, and the drop of gradients in backward |
